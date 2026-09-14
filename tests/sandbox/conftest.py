"""Shared fixtures for sandbox tests. No live E2B or Redis connection required."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.sandbox.session_registry import SessionRegistry


class FakeRedis:
    """Minimal in-memory stand-in for redis.asyncio.Redis, covering only
    the operations SessionRegistry actually uses."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    async def get(self, key: str) -> bytes | None:
        val = self._store.get(key)
        return val.encode() if val is not None else None

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def sadd(self, key: str, member: str) -> None:
        self._sets.setdefault(key, set()).add(member)

    async def srem(self, key: str, member: str) -> None:
        self._sets.get(key, set()).discard(member)

    async def smembers(self, key: str) -> set[bytes]:
        return {m.encode() for m in self._sets.get(key, set())}


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def session_registry(fake_redis: FakeRedis) -> SessionRegistry:
    return SessionRegistry(fake_redis)  # type: ignore[arg-type]


def make_fake_e2b_sandbox(
    sandbox_id: str = "fake-sandbox-id",
    stdout_lines: list[str] | None = None,
    stderr_lines: list[str] | None = None,
    error: Any = None,
    shell_exit_code: int = 0,
    shell_stdout: str = "",
    shell_stderr: str = "",
) -> SimpleNamespace:
    """
    Build an object matching the small slice of the E2B AsyncSandbox API
    that e2b_backend.py touches, so we can unit-test the backend logic
    without network access or real credentials.
    """
    run_code_result = SimpleNamespace(
        logs=SimpleNamespace(
            stdout=stdout_lines or [],
            stderr=stderr_lines or [],
        ),
        error=error,
    )
    shell_result = SimpleNamespace(
        exit_code=shell_exit_code, stdout=shell_stdout, stderr=shell_stderr
    )

    sandbox = SimpleNamespace(
        sandbox_id=sandbox_id,
        run_code=AsyncMock(return_value=run_code_result),
        commands=SimpleNamespace(run=AsyncMock(return_value=shell_result)),
        files=SimpleNamespace(
            write=AsyncMock(return_value=None),
            read=AsyncMock(return_value=b"file-content"),
        ),
        kill=AsyncMock(return_value=None),
    )
    return sandbox
