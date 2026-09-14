"""
LangGraph tool nodes wrapping SandboxManager.

Per PROJECT_CONTEXT.md, the orchestrator (and by extension the LLM's
function-calling surface) never talks to E2B/Docker directly — it only ever
sees these tool schemas, which delegate to whatever `SandboxManager`
instance the orchestrator was constructed with. Swapping E2B for Docker
later means changing one constructor call in `src/orchestrator/agent.py`,
not these tool definitions.

Each tool:
  - Takes the LangGraph-managed `thread_id` implicitly via closure, not as
    an LLM-visible argument (the model shouldn't be choosing session scope).
  - Returns a plain string summary for the LLM, never a raw ExecutionResult
    object, per your "structured JSON reasoning outputs" / clean schema
    requirement — the full ExecutionResult is logged for observability but
    not stuffed into the model's context unless truncated stdout/stderr is
    the useful payload.
"""

from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.sandbox.manager import SandboxError, SandboxManager, SessionNotFoundError
from src.sandbox.models import ExecutionRequest, Runtime

logger = logging.getLogger(__name__)


def _format_result_for_llm(result_success: bool, stdout: str, stderr: str, error_message: str | None) -> str:
    if error_message:
        return f"EXECUTION FAILED: {error_message}"
    parts = [f"exit_status: {'success' if result_success else 'failure'}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)


class RunCodeInput(BaseModel):
    code: str = Field(..., description="Source code to execute in the sandbox.")


class RunShellInput(BaseModel):
    command: str = Field(..., description="Shell command to run inside the sandbox (never on host).")


class CloneRepoInput(BaseModel):
    repo_url: str = Field(..., description="Git URL to clone (https:// or git@ only).")
    ref: str | None = Field(default=None, description="Branch, tag, or commit to check out.")


def build_sandbox_tools(
    manager: SandboxManager,
    session_id: str,
) -> list[StructuredTool]:
    """
    Build the set of LangGraph-compatible tools bound to a single, already
    provisioned sandbox session.

    Session provisioning/reuse (create_session / get_session against the
    Redis-backed registry) happens in the orchestrator graph node that runs
    *before* tool selection — these tool functions assume `session_id` is
    already valid for the current thread.
    """

    async def run_code(code: str) -> str:
        try:
            request = ExecutionRequest(session_id=session_id, kind="code", code=code)
            result = await manager.execute(request)
        except SandboxError as exc:
            logger.error("run_code tool failed session_id=%s: %s", session_id, exc)
            return f"EXECUTION FAILED: {exc}"
        return _format_result_for_llm(result.success, result.stdout, result.stderr, result.error_message)

    async def run_shell(command: str) -> str:
        try:
            request = ExecutionRequest(session_id=session_id, kind="shell", command=command)
            result = await manager.execute(request)
        except SandboxError as exc:
            logger.error("run_shell tool failed session_id=%s: %s", session_id, exc)
            return f"EXECUTION FAILED: {exc}"
        return _format_result_for_llm(result.success, result.stdout, result.stderr, result.error_message)

    async def clone_repo(repo_url: str, ref: str | None = None) -> str:
        try:
            request = ExecutionRequest(
                session_id=session_id, kind="clone_repo", repo_url=repo_url, ref=ref
            )
            result = await manager.execute(request)
        except (SandboxError, ValueError) as exc:
            # ValueError covers Pydantic's repo_url scheme validation in models.py
            logger.error("clone_repo tool failed session_id=%s: %s", session_id, exc)
            return f"EXECUTION FAILED: {exc}"
        return _format_result_for_llm(result.success, result.stdout, result.stderr, result.error_message)

    return [
        StructuredTool.from_function(
            coroutine=run_code,
            name="sandbox_run_code",
            description=(
                "Execute Python or Node code inside the isolated sandbox session for this "
                "task. Use for computation, data processing, or verifying logic. Never "
                "executes on the host system."
            ),
            args_schema=RunCodeInput,
        ),
        StructuredTool.from_function(
            coroutine=run_shell,
            name="sandbox_run_shell",
            description=(
                "Run a shell command inside the isolated sandbox filesystem for this task "
                "(e.g. installing a package, listing files). Never executes on the host system."
            ),
            args_schema=RunShellInput,
        ),
        StructuredTool.from_function(
            coroutine=clone_repo,
            name="sandbox_clone_repo",
            description=(
                "Clone a git repository into the sandbox filesystem for this task, so its "
                "contents can be inspected or run. Runs entirely inside the sandbox."
            ),
            args_schema=CloneRepoInput,
        ),
    ]


async def ensure_session_for_thread(
    manager: SandboxManager,
    thread_id: str,
    runtime: Runtime = Runtime.PYTHON,
    ttl_seconds: int = 600,
    existing_session_id: str | None = None,
) -> str:
    """
    Orchestrator-side helper: reuse the sandbox already bound to this
    LangGraph thread if one exists and is still live, otherwise provision a
    new one. Call this from the graph node that runs before tool selection,
    and store the returned session_id back into LangGraph state.
    """
    if existing_session_id:
        try:
            session = await manager.get_session(existing_session_id)
            if not session.is_expired():
                return existing_session_id
        except SessionNotFoundError:
            pass  # fall through to provisioning a fresh session

    session = await manager.create_session(
        runtime=runtime, ttl_seconds=ttl_seconds, thread_id=thread_id
    )
    return session.session_id
