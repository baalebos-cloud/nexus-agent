"""
E2B-backed implementation of SandboxManager.

This is the only file in the codebase that imports `e2b_code_interpreter`
directly. If we later add `docker_backend.py`, the orchestrator and all
LangGraph tool nodes remain unchanged because they depend only on
`SandboxManager` (manager.py).
"""

from __future__ import annotations

import logging
import time

from e2b_code_interpreter import AsyncSandbox

from src.sandbox.manager import SandboxError, SandboxManager, SessionNotFoundError
from src.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    Runtime,
    SandboxSession,
    SandboxBackend,
    SandboxStatus,
    SandboxPolicy,
)
from src.sandbox.policy import DEFAULT_POLICY, enforce_request, enforce_result
from src.sandbox.session_registry import SessionRegistry

logger = logging.getLogger(__name__)

_RUNTIME_TEMPLATE: dict[Runtime, str] = {
    Runtime.PYTHON: "base",
    Runtime.NODE: "base",
}


class E2BSandboxManager(SandboxManager):
    """SandboxManager implementation backed by the E2B Sandbox API."""

    def __init__(
        self,
        registry: SessionRegistry,
        policy: SandboxPolicy = DEFAULT_POLICY,
        e2b_api_key: str | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._api_key = e2b_api_key
        # Live E2B AsyncSandbox handles keyed by our session_id, held only for
        # the lifetime of this process — the durable record of truth is Redis.
        self._live_handles: dict[str, AsyncSandbox] = {}

    async def create_session(
        self,
        runtime: Runtime,
        ttl_seconds: int = 600,
        thread_id: str | None = None,
    ) -> SandboxSession:
        try:
            handle = await AsyncSandbox.create(
                template=_RUNTIME_TEMPLATE[runtime],
                api_key=self._api_key,
                timeout=ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - normalized into SandboxError below
            logger.error("E2B sandbox provisioning failed: %s", exc)
            raise SandboxError(f"Failed to provision E2B sandbox: {exc}") from exc

        session = SandboxSession(
            backend=SandboxBackend.E2B,
            backend_sandbox_id=handle.sandbox_id,
            runtime=runtime,
            status=SandboxStatus.READY,
            ttl_seconds=ttl_seconds,
            thread_id=thread_id,
        )
        self._live_handles[session.session_id] = handle
        await self._registry.save(session)
        logger.info(
            "Created E2B session session_id=%s sandbox_id=%s runtime=%s",
            session.session_id,
            handle.sandbox_id,
            runtime.value,
        )
        return session

    async def get_session(self, session_id: str) -> SandboxSession:
        return await self._registry.get(session_id)

    async def _get_live_handle(self, session_id: str) -> AsyncSandbox:
        handle = self._live_handles.get(session_id)
        if handle is not None:
            return handle

        # Process restarted or different worker handling this call — reattach
        # to the existing E2B sandbox by its provider-native ID.
        session = await self._registry.get(session_id)
        if session.status in (SandboxStatus.EXPIRED, SandboxStatus.CLOSED):
            raise SessionNotFoundError(f"Session {session_id!r} is {session.status.value}.")

        try:
            handle = await AsyncSandbox.connect(
                session.backend_sandbox_id, api_key=self._api_key
            )
        except Exception as exc:  # noqa: BLE001
            raise SessionNotFoundError(
                f"Could not reattach to E2B sandbox for session {session_id!r}: {exc}"
            ) from exc

        self._live_handles[session_id] = handle
        return handle

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        start = time.monotonic()

        try:
            request = enforce_request(request, self._policy)
        except SandboxError as exc:
            # PolicyViolation (and any other SandboxError) is a normal,
            # expected outcome here — not an application bug — so it's
            # normalized into a failed ExecutionResult rather than raised.
            return ExecutionResult(
                session_id=request.session_id,
                success=False,
                error_message=str(exc),
            )

        try:
            handle = await self._get_live_handle(request.session_id)
        except SessionNotFoundError as exc:
            return ExecutionResult(
                session_id=request.session_id,
                success=False,
                error_message=str(exc),
            )

        try:
            if request.kind == "code":
                result = await self._execute_code(handle, request)
            elif request.kind == "shell":
                result = await self._execute_shell(handle, request)
            elif request.kind == "clone_repo":
                result = await self._execute_clone(handle, request)
            else:  # exhaustive guard — Pydantic Literal should prevent this
                raise SandboxError(f"Unknown execution kind: {request.kind!r}")
        except Exception as exc:  # noqa: BLE001 - this is the normalization boundary
            logger.warning(
                "E2B execution failed session_id=%s kind=%s error=%s",
                request.session_id,
                request.kind,
                exc,
            )
            result = ExecutionResult(
                session_id=request.session_id,
                success=False,
                error_message=str(exc),
            )

        result.duration_ms = int((time.monotonic() - start) * 1000)
        return enforce_result(result, self._policy)

    async def _execute_code(
        self, handle: AsyncSandbox, request: ExecutionRequest
    ) -> ExecutionResult:
        if not request.code:
            return ExecutionResult(
                session_id=request.session_id,
                success=False,
                error_message="ExecutionRequest.kind='code' requires 'code' to be set.",
            )

        execution = await handle.run_code(
            request.code, timeout=request.timeout_seconds
        )
        stdout = "\n".join(execution.logs.stdout)
        stderr = "\n".join(execution.logs.stderr)
        has_error = execution.error is not None
        if has_error:
            stderr = f"{execution.error.name}: {execution.error.value}\n{stderr}"

        return ExecutionResult(
            session_id=request.session_id,
            success=not has_error,
            exit_code=0 if not has_error else 1,
            stdout=stdout,
            stderr=stderr,
        )

    async def _execute_shell(
        self, handle: AsyncSandbox, request: ExecutionRequest
    ) -> ExecutionResult:
        if not request.command:
            return ExecutionResult(
                session_id=request.session_id,
                success=False,
                error_message="ExecutionRequest.kind='shell' requires 'command' to be set.",
            )

        proc = await handle.commands.run(
            request.command, timeout=request.timeout_seconds
        )
        return ExecutionResult(
            session_id=request.session_id,
            success=proc.exit_code == 0,
            exit_code=proc.exit_code,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    async def _execute_clone(
        self, handle: AsyncSandbox, request: ExecutionRequest
    ) -> ExecutionResult:
        if not request.repo_url:
            return ExecutionResult(
                session_id=request.session_id,
                success=False,
                error_message="ExecutionRequest.kind='clone_repo' requires 'repo_url'.",
            )

        # git clone runs inside the E2B sandbox filesystem exclusively —
        # this process never shells out on the host, per Rule 2.
        clone_cmd = f"git clone --depth 1 {request.repo_url} /home/user/repo"
        if request.ref:
            clone_cmd += f" --branch {request.ref}"

        proc = await handle.commands.run(clone_cmd, timeout=request.timeout_seconds)
        return ExecutionResult(
            session_id=request.session_id,
            success=proc.exit_code == 0,
            exit_code=proc.exit_code,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    async def upload_file(
        self, session_id: str, dest_path: str, content: bytes
    ) -> ExecutionResult:
        try:
            handle = await self._get_live_handle(session_id)
            await handle.files.write(dest_path, content)
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                session_id=session_id, success=False, error_message=str(exc)
            )
        return ExecutionResult(session_id=session_id, success=True)

    async def download_file(self, session_id: str, src_path: str) -> bytes:
        handle = await self._get_live_handle(session_id)
        content = await handle.files.read(src_path)
        return content if isinstance(content, bytes) else content.encode("utf-8")

    async def close_session(self, session_id: str) -> None:
        handle = self._live_handles.pop(session_id, None)
        if handle is not None:
            try:
                await handle.kill()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error killing E2B sandbox session_id=%s: %s", session_id, exc)

        try:
            session = await self._registry.get(session_id)
            session = session.model_copy(update={"status": SandboxStatus.CLOSED})
            await self._registry.save(session)
        except SessionNotFoundError:
            pass
        finally:
            await self._registry.delete(session_id)

    async def reap_expired_sessions(self) -> list[str]:
        reaped: list[str] = []
        for session in await self._registry.list_all():
            if session.is_expired():
                logger.info("Reaping expired sandbox session_id=%s", session.session_id)
                await self.close_session(session.session_id)
                reaped.append(session.session_id)
        return reaped
