"""
Full-app integration tests, run against a real (local) Redis instance —
not mocked — to verify the FastAPI lifespan actually wires up correctly
and routes degrade gracefully (503, not a crash) when optional
dependencies (chat LLM, OSINT face detector) aren't configured.

These tests force chat_llm/video_processor/face_recognizer to an
unconfigured state on app.state AFTER the real lifespan runs, regardless
of what's actually in the local .env — otherwise this suite's pass/fail
would depend on whatever secrets happen to be present on the machine
running it, which is exactly the kind of non-determinism a test suite
should never have. Live-credential behavior is covered separately in
test_live_chat.py, opt-in only.

Requires a Redis instance reachable at the configured REDIS_URL (defaults
to redis://localhost:6379/0). Skipped automatically if unavailable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _redis_available() -> bool:
    import redis as sync_redis

    try:
        client = sync_redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        return client.ping()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="Redis not reachable at localhost:6379")


@pytest.fixture
def client():
    from src.api.server import app

    with TestClient(app) as c:
        # Force the "nothing optional configured" state deterministically,
        # independent of whatever the local .env actually contains.
        app.state.chat_llm = None
        app.state.video_processor = None
        app.state.face_recognizer = None
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_503_without_llm_configured(client):
    resp = client.post("/chat", json={"thread_id": "t1", "message": "hi"})
    assert resp.status_code == 503
    assert "chat_llm" in resp.json()["detail"]


def test_sandbox_session_lookup_returns_404_for_unknown_id(client):
    resp = client.get("/sandbox/sessions/does-not-exist")
    assert resp.status_code == 404


def test_osint_analyze_returns_503_without_face_detector(client):
    resp = client.post(
        "/osint/analyze-video",
        json={
            "video_path": "x.mp4",
            "purpose": "owned_footage_review",
            "requested_by": "alice",
            "justification": "testing the pipeline end to end",
        },
    )
    assert resp.status_code == 503
    assert "face detector" in resp.json()["detail"]


def test_osint_analyze_rejects_invalid_scope_with_400_before_checking_pipeline(client):
    """Law-enforcement purpose without case_reference must fail scope
    validation (400) even when the pipeline itself isn't configured —
    proves validation ordering puts input correctness before availability."""
    resp = client.post(
        "/osint/analyze-video",
        json={
            "video_path": "x.mp4",
            "purpose": "law_enforcement_request",
            "requested_by": "bob",
            "justification": "case work requested by department",
            "case_reference": None,
        },
    )
    assert resp.status_code == 400
