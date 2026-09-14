from __future__ import annotations

import pytest

from src.sandbox.models import ExecutionRequest, ExecutionResult, SandboxPolicy
from src.sandbox.policy import PolicyViolation, enforce_request, enforce_result


def test_enforce_request_clamps_oversized_timeout():
    policy = SandboxPolicy(max_timeout_seconds=60)
    req = ExecutionRequest(session_id="s1", kind="code", code="1+1", timeout_seconds=60)
    # timeout_seconds is capped at the model level (le=300) so use a valid-but-large value
    req = req.model_copy(update={"timeout_seconds": 200})
    result = enforce_request(req, policy)
    assert result.timeout_seconds == 60


def test_enforce_request_blocks_shell_when_disabled():
    policy = SandboxPolicy(allow_shell=False)
    req = ExecutionRequest(session_id="s1", kind="shell", command="ls")
    with pytest.raises(PolicyViolation):
        enforce_request(req, policy)


def test_enforce_request_blocks_clone_when_disabled():
    policy = SandboxPolicy(allow_clone_repo=False)
    req = ExecutionRequest(session_id="s1", kind="clone_repo", repo_url="https://github.com/x/y")
    with pytest.raises(PolicyViolation):
        enforce_request(req, policy)


def test_enforce_request_allows_code_regardless_of_shell_flag():
    policy = SandboxPolicy(allow_shell=False, allow_clone_repo=False)
    req = ExecutionRequest(session_id="s1", kind="code", code="print(1)")
    result = enforce_request(req, policy)
    assert result.kind == "code"


def test_enforce_result_truncates_long_stdout():
    policy = SandboxPolicy(max_output_chars=500)
    result = ExecutionResult(session_id="s1", success=True, stdout="x" * 1000)
    enforced = enforce_result(result, policy)
    assert len(enforced.stdout) <= 500 + len("\n...[truncated by policy]...")
    assert enforced.truncated is True


def test_enforce_result_leaves_short_output_untouched():
    policy = SandboxPolicy(max_output_chars=1000)
    result = ExecutionResult(session_id="s1", success=True, stdout="short")
    enforced = enforce_result(result, policy)
    assert enforced.stdout == "short"
    assert enforced.truncated is False


def test_repo_url_rejects_non_git_scheme():
    with pytest.raises(ValueError):
        ExecutionRequest(session_id="s1", kind="clone_repo", repo_url="file:///etc/passwd")
