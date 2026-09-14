"""
Integration test for the /chat -> capture_conversation wiring in
src/api/routes.py. Uses a fake LLM (no real Groq/E2B credentials needed)
but exercises the REAL route, REAL graph, and REAL capture module —
proving the wiring itself works, not just each piece in isolation.

Requires Redis, like the other server integration tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api.config import Settings
from src.finetune.data_schema import TrainingExample
from src.sandbox.models import Runtime, SandboxBackend, SandboxSession


def _redis_available() -> bool:
    import redis as sync_redis

    try:
        client = sync_redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        return client.ping()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="Redis not reachable at localhost:6379")


class FakeLLM:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="This is a fake reply.")


def _fake_sandbox_manager() -> AsyncMock:
    """
    server.py's lifespan wires a real, E2B-backed sandbox_manager onto
    app.state — which needs a real E2B_API_KEY to provision anything.
    These tests care about the /chat -> capture wiring, not sandbox
    provisioning, so app.state.sandbox_manager is swapped for a fake here
    (mirroring tests/orchestrator/test_agent_graph.py's approach) rather
    than requiring real E2B credentials just to exercise capture.
    """
    manager = AsyncMock()
    manager.create_session = AsyncMock(
        return_value=SandboxSession(
            session_id="fake-sess-1",
            backend=SandboxBackend.E2B,
            backend_sandbox_id="fake-sbx-1",
            runtime=Runtime.PYTHON,
        )
    )
    return manager


def test_chat_route_captures_conversation_when_enabled(tmp_path):
    capture_path = str(tmp_path / "captured.jsonl")
    pending_path = str(tmp_path / "pending.jsonl")

    test_settings = Settings(capture_training_data=True)
    test_settings = test_settings.model_copy(
        update={
            "training_capture_path": capture_path,
            "training_capture_pending_review_path": pending_path,
        }
    )

    with patch("src.api.routes.get_settings", return_value=test_settings):
        from src.api.server import app

        with TestClient(app) as client:
            app.state.chat_llm = FakeLLM()  # bypass the real-LLM requirement
            app.state.sandbox_manager = _fake_sandbox_manager()
            app.state.video_processor = None
            app.state.face_recognizer = None

            resp = client.post("/chat", json={"thread_id": "capture-test-1", "message": "hi there"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["reply"] == "This is a fake reply."

    assert Path(capture_path).exists(), "Expected a training example to have been captured."
    with open(capture_path) as f:
        record = json.loads(f.readline())

    example = TrainingExample.model_validate(record)
    assert example.source == "live_capture_sandbox"
    assert example.turns[-1].content == "This is a fake reply."


def test_chat_route_does_not_capture_when_disabled(tmp_path):
    capture_path = str(tmp_path / "captured.jsonl")
    pending_path = str(tmp_path / "pending.jsonl")

    test_settings = Settings(capture_training_data=False)  # explicit default
    test_settings = test_settings.model_copy(
        update={
            "training_capture_path": capture_path,
            "training_capture_pending_review_path": pending_path,
        }
    )

    with patch("src.api.routes.get_settings", return_value=test_settings):
        from src.api.server import app

        with TestClient(app) as client:
            app.state.chat_llm = FakeLLM()
            app.state.sandbox_manager = _fake_sandbox_manager()
            app.state.video_processor = None
            app.state.face_recognizer = None

            resp = client.post("/chat", json={"thread_id": "capture-test-2", "message": "hi there"})

    assert resp.status_code == 200
    assert not Path(capture_path).exists(), "Capture must be a true no-op when disabled."
