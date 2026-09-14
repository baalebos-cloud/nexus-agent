"""
SandboxManager — the single interface the orchestrator is allowed to import
for code/shell execution.

Per PROJECT_CONTEXT.md Rule 2 (Security & Execution Safety):
    - No `git clone`, `eval()`, or arbitrary bash on the host OS, ever.
    - All execution routes through this interface into a containerized or
      virtualized backend (E2B today, Docker later).

Backend implementations (e2b_backend.py, docker_backend.py) subclass
`SandboxManager` and are swappable without touching orchestrator code,
because the orchestrator only ever type-hints against this ABC.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    Runtime,
    SandboxSession,
)

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Base exception for sandbox-layer failures that could not be normalized
    into an ExecutionResult (e.g. failure to provision a session at all)."""


class SessionNotFoundError(SandboxError):
    """Raised when an operation references a session_id that doesn't exist
    or has already expired/closed."""


class SandboxManager(ABC):
    """
    Abstract sandbox execution interface.

    Implementations MUST:
      - Never execute anything outside the provider's isolated sandbox.
      - Never raise raw provider exceptions across this boundary — catch them
        and return a `ExecutionResult(success=False, error_message=...)`
        instead, except for session-lifecycle failures (create/close), which
        may raise `SandboxError` subclasses since there's no ExecutionResult
        to attach them to.
      - Enforce timeouts and output caps via `policy.py`, not ad hoc.
    """

    @abstractmethod
    async def create_session(
        self,
        runtime: Runtime,
        ttl_seconds: int = 600,
        thread_id: str | None = None,
    ) -> SandboxSession:
        """Provision a new sandbox session and register it (e.g. in Redis)."""
        raise NotImplementedError

    @abstractmethod
    async def get_session(self, session_id: str) -> SandboxSession:
        """Fetch session metadata. Raises SessionNotFoundError if absent/expired."""
        raise NotImplementedError

    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute code inside an existing session.

        `request.kind` dispatches to the appropriate sandbox operation
        (code / shell / clone_repo). Implementations should route all three
        through this single entrypoint so policy enforcement happens in one
        place.
        """
        raise NotImplementedError

    @abstractmethod
    async def upload_file(
        self, session_id: str, dest_path: str, content: bytes
    ) -> ExecutionResult:
        """Write a file into the sandbox filesystem."""
        raise NotImplementedError

    @abstractmethod
    async def download_file(self, session_id: str, src_path: str) -> bytes:
        """Read a file out of the sandbox filesystem."""
        raise NotImplementedError

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Tear down the sandbox and remove it from the session registry."""
        raise NotImplementedError

    @abstractmethod
    async def reap_expired_sessions(self) -> list[str]:
        """
        Find sessions past their TTL and close them.

        Called by a background task, not solely relied on for cleanup —
        provider-side timeouts are a backstop, not the primary mechanism,
        since we don't want to leak billable sandbox-hours.

        Returns the list of session_ids that were reaped.
        """
        raise NotImplementedError
