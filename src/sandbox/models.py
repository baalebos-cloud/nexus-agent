"""
Data models for the Sandboxed Execution Engine.

All cross-boundary data (orchestrator <-> SandboxManager <-> backend) flows
through these Pydantic v2 models. Nothing outside this module should define
ad-hoc dicts for execution requests/results — this is the single source of
truth for the execution contract referenced in PROJECT_CONTEXT.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Runtime(str, Enum):
    """Supported sandbox runtimes."""

    PYTHON = "python"
    NODE = "node"


class SandboxBackend(str, Enum):
    """Supported sandbox execution backends."""

    E2B = "e2b"
    DOCKER = "docker"


class SandboxStatus(str, Enum):
    """Lifecycle status of a sandbox session."""

    PROVISIONING = "provisioning"
    READY = "ready"
    EXECUTING = "executing"
    IDLE = "idle"
    EXPIRED = "expired"
    CLOSED = "closed"
    ERROR = "error"


class SandboxSession(BaseModel):
    """
    Represents a live (or recently live) sandbox session.

    Persisted in Redis as `session:{session_id}` so long-horizon LangGraph
    tasks can reuse the same sandbox across multiple tool calls instead of
    re-provisioning on every step.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    backend: SandboxBackend
    backend_sandbox_id: str = Field(
        ..., description="Provider-native sandbox/session ID (e.g. E2B sandbox ID)."
    )
    runtime: Runtime
    status: SandboxStatus = SandboxStatus.PROVISIONING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = Field(default=600, ge=30, le=3600)
    thread_id: str | None = Field(
        default=None,
        description="LangGraph conversation/task thread this session is bound to.",
    )
    secrets_injected: bool = Field(
        default=False,
        description="Whether this session has explicit opt-in secret/credential injection.",
    )

    def is_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self.last_active_at).total_seconds()
        return elapsed > self.ttl_seconds


class ExecutionRequest(BaseModel):
    """A single unit of work to run inside an existing sandbox session."""

    session_id: str
    kind: Literal["code", "shell", "clone_repo"]
    code: str | None = Field(default=None, description="Source code to execute.")
    command: str | None = Field(default=None, description="Shell command to run inside the sandbox.")
    repo_url: str | None = Field(default=None, description="Git URL to clone inside the sandbox.")
    ref: str | None = Field(default=None, description="Branch, tag, or commit to check out.")
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    @field_validator("repo_url")
    @classmethod
    def _validate_repo_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (v.startswith("https://") or v.startswith("git@")):
            raise ValueError("repo_url must use https:// or git@ (no local/file paths).")
        return v


class ExecutionResult(BaseModel):
    """
    Normalized result of any execution call.

    The orchestrator only ever sees this shape — never a raw provider
    exception or an unstructured stack trace. Callers in manager.py are
    responsible for catching backend-specific exceptions and mapping them
    into `success=False` results with `error_message` populated.
    """

    session_id: str
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    error_message: str | None = None
    duration_ms: int | None = None

    @field_validator("stdout", "stderr")
    @classmethod
    def _cap_output_length(cls, v: str) -> str:
        # Hard backstop cap; policy.py enforces the tighter, configurable limit.
        # This exists so a misbehaving backend can never blow up the LLM context.
        max_chars = 50_000
        if len(v) > max_chars:
            return v[:max_chars] + "\n...[truncated at model boundary]..."
        return v


class SandboxPolicy(BaseModel):
    """Enforcement parameters applied by policy.py before any execution runs."""

    default_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_timeout_seconds: int = Field(default=120, ge=1, le=600)
    max_output_chars: int = Field(default=8_000, ge=500)
    allow_shell: bool = True
    allow_clone_repo: bool = True
    allow_secret_injection: bool = False
