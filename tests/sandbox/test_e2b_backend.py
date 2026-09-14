from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.sandbox.e2b_backend import E2BSandboxManager
from src.sandbox.manager import SessionNotFoundError
from src.sandbox.models import ExecutionRequest, Runtime, SandboxPolicy
from tests.sandbox.conftest import make_fake_e2b_sandbox


@pytest.mark.asyncio
async def test_create_session_registers_in_redis(session_registry):
    fake_sandbox = make_fake_e2b_sandbox(sandbox_id="sbx-1")
    with patch(
        "src.sandbox.e2b_backend.AsyncSandbox.create",
        new=AsyncMock(return_value=fake_sandbox),
    ):
        manager = E2BSandboxManager(registry=session_registry)
        session = await manager.create_session(runtime=Runtime.PYTHON, ttl_seconds=300)

    assert session.backend_sandbox_id == "sbx-1"
    stored = await session_registry.get(session.session_id)
    assert stored.session_id == session.session_id


@pytest.mark.asyncio
async def test_execute_code_success(session_registry):
    fake_sandbox = make_fake_e2b_sandbox(stdout_lines=["hello"], stderr_lines=[])
    with patch(
        "src.sandbox.e2b_backend.AsyncSandbox.create",
        new=AsyncMock(return_value=fake_sandbox),
    ):
        manager = E2BSandboxManager(registry=session_registry)
        session = await manager.create_session(runtime=Runtime.PYTHON)

    request = ExecutionRequest(session_id=session.session_id, kind="code", code="print('hello')")
    result = await manager.execute(request)

    assert result.success is True
    assert result.stdout == "hello"
    assert result.duration_ms is not None


@pytest.mark.asyncio
async def test_execute_code_with_error_reports_failure(session_registry):
    fake_error = type("E2BError", (), {"name": "ValueError", "value": "bad input"})()
    fake_sandbox = make_fake_e2b_sandbox(stdout_lines=[], stderr_lines=["trace"], error=fake_error)
    with patch(
        "src.sandbox.e2b_backend.AsyncSandbox.create",
        new=AsyncMock(return_value=fake_sandbox),
    ):
        manager = E2BSandboxManager(registry=session_registry)
        session = await manager.create_session(runtime=Runtime.PYTHON)

    request = ExecutionRequest(session_id=session.session_id, kind="code", code="raise ValueError")
    result = await manager.execute(request)

    assert result.success is False
    assert "ValueError" in result.stderr


@pytest.mark.asyncio
async def test_execute_shell_runs_inside_sandbox_only(session_registry):
    fake_sandbox = make_fake_e2b_sandbox(shell_exit_code=0, shell_stdout="file1\nfile2")
    with patch(
        "src.sandbox.e2b_backend.AsyncSandbox.create",
        new=AsyncMock(return_value=fake_sandbox),
    ):
        manager = E2BSandboxManager(registry=session_registry)
        session = await manager.create_session(runtime=Runtime.PYTHON)

    request = ExecutionRequest(session_id=session.session_id, kind="shell", command="ls")
    result = await manager.execute(request)

    assert result.success is True
    assert result.stdout == "file1\nfile2"
    fake_sandbox.commands.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_clone_repo_builds_correct_command(session_registry):
    fake_sandbox = make_fake_e2b_sandbox(shell_exit_code=0)
    with patch(
        "src.sandbox.e2b_backend.AsyncSandbox.create",
        new=AsyncMock(return_value=fake_sandbox),
    ):
        manager = E2BSandboxManager(registry=session_registry)
        session = await manager.create_session(runtime=Runtime.PYTHON)

    request = ExecutionRequest(
        session_id=session.session_id,
        kind="clone_repo",
        repo_url="https://github.com/anthropics/example.git",
        ref="main",
    )
    result = await manager.execute(request)

    assert result.success is True
    called_cmd = fake_sandbox.commands.run.call_args.args[0]
    assert "git clone --depth 1 https://github.com/anthropics/example.git" in called_cmd
    assert "--branch main" in called_cmd


@pytest.mark.asyncio
async def test_execute_against_unknown_session_returns_failed_result(session_registry):
    manager = E2BSandboxManager(registry=session_registry)
    request = ExecutionRequest(session_id="does-not-exist", kind="code", code="1+1")
    result = await manager.execute(request)

    assert result.success is False
    assert result.error_message is not None


@pytest.mark.asyncio
async def test_close_session_removes_from_registry(session_registry):
    fake_sandbox = make_fake_e2b_sandbox()
    with patch(
        "src.sandbox.e2b_backend.AsyncSandbox.create",
        new=AsyncMock(return_value=fake_sandbox),
    ):
        manager = E2BSandboxManager(registry=session_registry)
        session = await manager.create_session(runtime=Runtime.PYTHON)

    await manager.close_session(session.session_id)
    fake_sandbox.kill.assert_awaited_once()

    with pytest.raises(SessionNotFoundError):
        await session_registry.get(session.session_id)


@pytest.mark.asyncio
async def test_policy_blocks_shell_before_reaching_e2b(session_registry):
    fake_sandbox = make_fake_e2b_sandbox()
    with patch(
        "src.sandbox.e2b_backend.AsyncSandbox.create",
        new=AsyncMock(return_value=fake_sandbox),
    ):
        manager = E2BSandboxManager(
            registry=session_registry, policy=SandboxPolicy(allow_shell=False)
        )
        session = await manager.create_session(runtime=Runtime.PYTHON)

    request = ExecutionRequest(session_id=session.session_id, kind="shell", command="rm -rf /")
    result = await manager.execute(request)

    assert result.success is False
    assert "disabled by policy" in (result.error_message or "")
    fake_sandbox.commands.run.assert_not_awaited()
