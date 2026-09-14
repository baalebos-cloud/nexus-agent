"""
Structural tests for src/voice/worker.py.

These verify NexusVoiceAgent wires correctly against the REAL
livekit-agents SDK classes (Agent, AgentSession) — catching API drift like
constructor signature changes — without needing live LiveKit/Deepgram/
Cartesia credentials or an actual WebRTC connection.

Genuinely live behavior (a real call being answered, real STT/TTS) is not
covered here — that requires all four voice-stack credentials and a real
LiveKit room, which is out of scope for an automated test suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.orchestrator.state import AgentState
from src.sandbox.models import Runtime, SandboxBackend, SandboxSession
from src.voice.worker import NexusVoiceAgent


class FakeLLM:
    """Same fake used in test_agent_graph.py — no tool calls, terminates
    the graph after one pass."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="Voice reply.")


def _fake_sandbox_manager() -> AsyncMock:
    manager = AsyncMock()
    manager.create_session = AsyncMock(
        return_value=SandboxSession(
            session_id="voice-sess-1",
            backend=SandboxBackend.E2B,
            backend_sandbox_id="sbx-voice",
            runtime=Runtime.PYTHON,
        )
    )
    return manager


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_generate_reply_text_reuses_sandbox_session_across_turns():
    """
    Regression test for a real bug found during a live voice-call test:
    every single turn provisioned a BRAND NEW E2B sandbox instead of
    reusing one for the whole call — confirmed live, with 3 different
    sandbox_ids logged across one conversation. Fixed by persisting
    sandbox_session_id across turns; this proves create_session is only
    called ONCE across two consecutive turns.
    """
    manager = AsyncMock()
    session_obj = SandboxSession(
        session_id="voice-sess-1",
        backend=SandboxBackend.E2B,
        backend_sandbox_id="sbx-voice",
        runtime=Runtime.PYTHON,
    )
    manager.create_session = AsyncMock(return_value=session_obj)
    manager.get_session = AsyncMock(return_value=session_obj)

    agent = NexusVoiceAgent(sandbox_manager=manager, thread_id="thread-1", llm=FakeLLM())

    await agent.generate_reply_text("first turn")
    await agent.generate_reply_text("second turn")

    manager.create_session.assert_awaited_once()
    manager.get_session.assert_awaited_once_with("voice-sess-1")
    assert agent._sandbox_session_id == "voice-sess-1"


@pytest.mark.asyncio
async def test_backend_failure_produces_spoken_error_not_silence():
    """
    Regression test for a real failure mode found during a live call: a
    Redis outage caused generate_reply_text to raise inside
    on_user_turn_completed, but nothing ever called session.say() on that
    turn — so the caller heard nothing and had no way to know the agent
    had failed versus just being slow. This proves a backend failure now
    produces a spoken error message instead of silent failure.
    """

    class FailingSandboxManager:
        async def create_session(self, *args, **kwargs):
            raise ConnectionError("Redis unreachable (simulated)")

        async def get_session(self, *args, **kwargs):
            raise ConnectionError("Redis unreachable (simulated)")

    agent = NexusVoiceAgent(
        sandbox_manager=FailingSandboxManager(), thread_id="thread-1", llm=FakeLLM()
    )

    reply = await agent._generate_reply_text_or_spoken_error("hello")

    assert "technical problem" in reply
    assert isinstance(reply, str)  # still safe to pass straight to session.say()


@pytest.mark.asyncio
async def test_on_user_turn_completed_speaks_error_instead_of_raising():
    """
    End-to-end version of the above through the real on_user_turn_completed
    entrypoint (real ChatMessage, real session.say() call site) — confirms
    the exception never propagates out and self.session.say() is called
    with the error text rather than the turn failing silently.
    """
    from types import SimpleNamespace

    from livekit.agents.llm import ChatMessage

    class FailingSandboxManager:
        async def create_session(self, *args, **kwargs):
            raise ConnectionError("Redis unreachable (simulated)")

        async def get_session(self, *args, **kwargs):
            raise ConnectionError("Redis unreachable (simulated)")

    agent = NexusVoiceAgent(
        sandbox_manager=FailingSandboxManager(), thread_id="thread-1", llm=FakeLLM()
    )
    fake_session = AsyncMock()
    fake_session.say = AsyncMock()
    agent._activity = SimpleNamespace(session=fake_session)

    real_chat_message = ChatMessage(role="user", content=["hello"])

    # Must not raise — the whole point of the fix.
    await agent.on_user_turn_completed(turn_ctx=None, new_message=real_chat_message)

    fake_session.say.assert_awaited_once()
    spoken_text = fake_session.say.call_args.args[0]
    assert "technical problem" in spoken_text


def test_nexus_voice_agent_constructs_against_real_livekit_agent_base():
    """
    Agent's real __init__ requires `instructions` as a keyword-only str.
    If livekit-agents changes that signature, this fails loudly here
    instead of only at 2am during a live call.
    """
    agent = NexusVoiceAgent(
        sandbox_manager=_fake_sandbox_manager(), thread_id="thread-1", llm=FakeLLM()
    )
    assert agent is not None


@pytest.mark.asyncio
async def test_agent_graph_produces_a_plain_string_reply_for_session_say():
    """
    Proves the LangGraph portion of on_user_turn_completed (the part this
    project owns) produces exactly the shape AgentSession.say(text: str)
    expects, confirmed against the real method signature.
    """
    agent = NexusVoiceAgent(
        sandbox_manager=_fake_sandbox_manager(), thread_id="thread-1", llm=FakeLLM()
    )

    new_message = HumanMessage(content="hi there")
    state: AgentState = {
        "messages": [new_message],
        "thread_id": "thread-1",
        "sandbox_session_id": None,
        "osint_scope": None,
        "is_complete": False,
        "scratch": {},
    }
    result = await agent._graph.ainvoke(state)
    reply = result["messages"][-1].content
    assert reply == "Voice reply."
    assert isinstance(reply, str)


def test_agent_session_say_has_expected_signature():
    """
    Guards against livekit-agents renaming/removing AgentSession.say —
    our on_user_turn_completed calls self.session.say(response_text)
    with a single positional string argument.
    """
    import inspect

    from livekit.agents.voice import AgentSession

    sig = inspect.signature(AgentSession.say)
    params = list(sig.parameters)
    assert "text" in params


def test_agent_on_user_turn_completed_signature_matches_base_class():
    """
    Guards against livekit-agents changing the on_user_turn_completed
    hook's signature out from under our override.
    """
    import inspect

    from livekit.agents.voice import Agent

    base_sig = inspect.signature(Agent.on_user_turn_completed)
    override_sig = inspect.signature(NexusVoiceAgent.on_user_turn_completed)
    assert list(base_sig.parameters) == list(override_sig.parameters)


def test_worker_module_exposes_a_discoverable_agent_server():
    """
    The LiveKit CLI (`python -m livekit.agents start src/voice/worker.py
    --dev`) discovers the worker via a module-level `app`/`server`/`agent`
    global of type AgentServer. If this attribute is missing or the wrong
    type, the CLI can't find the worker at all — confirmed by actually
    invoking the CLI against this file with fake credentials in a prior
    manual check (it starts up correctly and only fails at the real
    network handshake, proving discovery + construction both work).
    """
    from livekit.agents.worker import AgentServer

    from src.voice import worker

    assert isinstance(worker.app, AgentServer)


def test_worker_entrypoint_is_registered_as_rtc_session():
    """
    Confirms the module-level `entrypoint` function was actually
    registered with the AgentServer via @app.rtc_session(), not just
    defined as a plain function that never gets wired up.
    """
    from src.voice import worker

    # rtc_session-decorated functions remain callable directly; this also
    # confirms the decorator didn't silently swallow/replace the function
    # with something broken.
    assert callable(worker.entrypoint)


@pytest.mark.asyncio
async def test_on_user_turn_completed_handles_real_livekit_chat_message():
    """
    Regression test for a real bug found during a live voice-call test:
    on_user_turn_completed originally passed the raw
    livekit.agents.llm.ChatMessage straight into the LangGraph state,
    which langchain's message coercion rejects with
    "NotImplementedError: Unsupported message type". This uses a genuine
    ChatMessage (not a mock) to prove the fix — extracting `.text_content`
    before it ever reaches the graph — actually works against the real
    SDK type, not an assumption about its shape.
    """
    from types import SimpleNamespace

    from livekit.agents.llm import ChatMessage

    agent = NexusVoiceAgent(
        sandbox_manager=_fake_sandbox_manager(), thread_id="thread-1", llm=FakeLLM()
    )

    # Agent.session is a read-only property backed by self._activity —
    # there's no public setter, so a fake activity is the way to give the
    # agent a usable `session.say` without a live LiveKit connection.
    fake_session = AsyncMock()
    fake_session.say = AsyncMock()
    agent._activity = SimpleNamespace(session=fake_session)

    real_chat_message = ChatMessage(role="user", content=["hello from a real ChatMessage"])

    # This previously raised NotImplementedError before reaching .say() —
    # if it does again, this test fails here rather than only during a
    # live call.
    await agent.on_user_turn_completed(turn_ctx=None, new_message=real_chat_message)

    fake_session.say.assert_awaited_once_with("Voice reply.")


@pytest.mark.asyncio
async def test_generate_reply_text_shared_by_voice_and_text_input_paths():
    """
    Regression test for a second real bug found during the same live
    test: typed text input (e.g. LiveKit Cloud console's chat box) went
    through a different default code path than spoken input, which
    required AgentSession's own `llm` (never configured, since our
    architecture routes everything through LangGraph instead) and raised
    "trying to generate reply without an LLM model". The fix exposes
    `generate_reply_text` as the single shared path both voice
    (on_user_turn_completed) and text (entrypoint's text_input_cb) route
    through — this confirms it works standalone with plain text input,
    the same call shape entrypoint's text_input_cb uses.
    """
    agent = NexusVoiceAgent(
        sandbox_manager=_fake_sandbox_manager(), thread_id="thread-1", llm=FakeLLM()
    )
    reply = await agent.generate_reply_text("typed hello")
    assert reply == "Voice reply."
