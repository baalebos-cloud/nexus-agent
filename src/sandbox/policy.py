"""
Execution safety policy — the enforcement point referenced in
PROJECT_CONTEXT.md Rule 2, not just documentation.

Backends call `enforce_request` before touching the provider API and
`enforce_result` before returning to the orchestrator, so limits are
guaranteed regardless of which backend is active.
"""

from __future__ import annotations

from src.sandbox.manager import SandboxError
from src.sandbox.models import ExecutionRequest, ExecutionResult, SandboxPolicy

DEFAULT_POLICY = SandboxPolicy()


class PolicyViolation(SandboxError):
    """Raised when a request violates the active SandboxPolicy."""


def enforce_request(
    request: ExecutionRequest, policy: SandboxPolicy = DEFAULT_POLICY
) -> ExecutionRequest:
    """
    Validate and clamp an ExecutionRequest against policy before dispatch.

    Raises PolicyViolation for outright-disallowed operations (e.g. shell
    access when `allow_shell=False`). Clamps timeout to the policy ceiling
    rather than rejecting, since an oversized timeout is a caller mistake,
    not a security issue.
    """
    if request.kind == "shell" and not policy.allow_shell:
        raise PolicyViolation("Shell execution is disabled by policy for this deployment.")

    if request.kind == "clone_repo" and not policy.allow_clone_repo:
        raise PolicyViolation("git clone is disabled by policy for this deployment.")

    if request.timeout_seconds > policy.max_timeout_seconds:
        request = request.model_copy(
            update={"timeout_seconds": policy.max_timeout_seconds}
        )

    return request


def enforce_result(
    result: ExecutionResult, policy: SandboxPolicy = DEFAULT_POLICY
) -> ExecutionResult:
    """Truncate stdout/stderr to the policy's max_output_chars before the
    result reaches the LLM context window."""
    updates: dict[str, object] = {}

    if len(result.stdout) > policy.max_output_chars:
        updates["stdout"] = (
            result.stdout[: policy.max_output_chars] + "\n...[truncated by policy]..."
        )
        updates["truncated"] = True

    if len(result.stderr) > policy.max_output_chars:
        updates["stderr"] = (
            result.stderr[: policy.max_output_chars] + "\n...[truncated by policy]..."
        )
        updates["truncated"] = True

    if not updates:
        return result

    return result.model_copy(update=updates)
