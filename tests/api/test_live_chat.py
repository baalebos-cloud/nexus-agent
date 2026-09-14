"""
Opt-in LIVE integration tests — these make real network calls to whatever
is configured in your actual .env (Groq/vLLM, E2B). Skipped by default.

Run explicitly with:

    RUN_LIVE_TESTS=1 pytest tests/api/test_live_chat.py -v

Unlike the rest of the suite, these are not mocked, not deterministic
(depend on external service availability/quotas), and may cost real
API credits. That's the correct tradeoff for a test whose entire purpose
is proving a live credential actually works end-to-end.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="Live tests are opt-in — set RUN_LIVE_TESTS=1 to run them against your real .env.",
)


@pytest.fixture
def client():
    from src.api.server import app

    with TestClient(app) as c:
        yield c


def test_chat_with_real_configured_llm_returns_a_reply(client):
    """
    If this fails with a 503, STREAMING_LLM_BASE_URL isn't set in .env.
    If this fails with a 500 wrapping an OpenAINotFoundError, the model ID
    in STREAMING_LLM_MODEL doesn't exist for your key — recheck against:
        curl -s https://api.groq.com/openai/v1/models \\
          -H "Authorization: Bearer $STREAMING_LLM_API_KEY"
    """
    resp = client.post(
        "/chat", json={"thread_id": "live-test-1", "message": "Reply with exactly the word: pong"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reply"], "Expected a non-empty reply from the live LLM."
    # sandbox_session_id should be populated too, proving E2B provisioning
    # succeeded as part of the same request (graph runs `provision` first).
    assert body["sandbox_session_id"], "Expected a real sandbox session id — check your E2B key."


def test_chat_llm_can_execute_sandboxed_code(client):
    """
    A stronger end-to-end check: asks the model to use the sandbox tool,
    not just reply in text. Proves the full chain — LLM tool-calling,
    E2B provisioning, code execution, and the result flowing back through
    the graph — actually works together.
    """
    resp = client.post(
        "/chat",
        json={
            "thread_id": "live-test-2",
            "message": (
                "Use your code execution tool to compute sum(range(100)) in Python "
                "and tell me only the resulting number."
            ),
        },
    )
    assert resp.status_code == 200, resp.text
    reply = resp.json()["reply"]
    assert "4950" in reply, f"Expected the correct sum (4950) in the reply, got: {reply!r}"
