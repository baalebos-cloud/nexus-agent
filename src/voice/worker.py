"""
Real-Time Voice Stack — LiveKit WebRTC agent worker.

Pipeline: Silero VAD -> Deepgram STT -> Nexus-Agent LangGraph orchestrator
(text in, text out) -> Cartesia TTS.

This worker deliberately reuses `build_agent_graph` from
src/orchestrator/agent.py rather than standing up a second, separate LLM
call path — voice and text share one orchestrator so tool access (sandbox,
OSINT-when-scoped) and system prompt behavior stay consistent regardless of
input modality.

Requires a running LiveKit server (self-hosted or LiveKit Cloud) and API
keys for Deepgram + Cartesia, sourced via src/api/config.py — this module
does not read os.environ directly.

IMPORTANT — API version note: an earlier version of this file used
`livekit.agents.cli.run_app(WorkerOptions(...))`. That path is broken in
current livekit-agents releases — `cli.run_app` internally imports a
`livekit.agents._legacy` submodule that no longer exists in the installed
package, so it would raise `ModuleNotFoundError` immediately on invocation.
This rewrite uses the current, actually-supported `AgentServer` +
`@app.rtc_session()` pattern instead, confirmed against the real installed
SDK before shipping.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from livekit.agents import AutoSubscribe, JobContext
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomOptions, TextInputOptions
from livekit.agents.worker import AgentServer
from livekit.plugins import cartesia, deepgram, silero

from src.api.config import get_settings
from src.api.llm import build_chat_llm
from src.finetune.capture import capture_conversation
from src.orchestrator.agent import build_agent_graph
from src.orchestrator.state import AgentState
from src.sandbox.e2b_backend import E2BSandboxManager
from src.sandbox.manager import SandboxManager
from src.sandbox.models import SandboxPolicy
from src.sandbox.session_registry import SessionRegistry

logger = logging.getLogger(__name__)

_settings = get_settings()


class NexusVoiceAgent(Agent):
    """
    Bridges LiveKit's turn-based voice session to the LangGraph orchestrator.

    LiveKit's AgentSession handles VAD-based turn detection and STT/TTS
    plumbing; `on_user_turn_completed` is where a completed user utterance
    is handed to the LangGraph graph and the resulting text response is
    sent back for TTS synthesis.
    """

    def __init__(self, sandbox_manager: SandboxManager, thread_id: str, llm) -> None:
        super().__init__(instructions="")  # system prompt lives in agent.py's SYSTEM_PROMPT
        self._graph = build_agent_graph(
            llm=llm,
            sandbox_manager=sandbox_manager,
            # Voice sessions default to no OSINT scope: identity-matching
            # over a live video/audio call is exactly the kind of
            # unscoped, ambient-surveillance use case the AnalysisScope
            # gate exists to prevent. A scope can be attached explicitly
            # by whatever authenticated call-handling flow provisions this
            # agent, but it is never the default.
            osint_scope=None,
        )
        self._thread_id = thread_id
        # Persisted across turns within this call so the orchestrator's
        # session-reuse logic (ensure_session_for_thread, in
        # sandbox_tools.py) actually gets to do its job. Without this, a
        # live call was observed provisioning a BRAND NEW E2B sandbox on
        # every single turn (confirmed in a real test: 3 different
        # sandbox_ids logged across one conversation) instead of reusing
        # one sandbox for the whole call — wasted, billable sandbox-hours
        # for no reason.
        self._sandbox_session_id: str | None = None

    async def generate_reply_text(self, user_text: str) -> str:
        """
        Runs a single user utterance through the LangGraph orchestrator and
        returns the reply text. Shared by both the voice turn-completion
        path (on_user_turn_completed, below) and the text-chat input path
        (wired up in entrypoint()'s RoomInputOptions.text_input_cb) so
        typed and spoken input get identical handling — same tools, same
        system prompt, same everything — rather than diverging behavior
        depending on input modality.
        """
        state: AgentState = {
            "messages": [HumanMessage(content=user_text)],
            "thread_id": self._thread_id,
            "sandbox_session_id": self._sandbox_session_id,
            "osint_scope": None,
            "is_complete": False,
            "scratch": {},
        }
        result = await self._graph.ainvoke(state)
        # Persist whatever session the graph provisioned/reused this turn,
        # so the NEXT turn passes it back in above instead of leaving it
        # None (which is what silently forced a fresh sandbox every time).
        self._sandbox_session_id = result.get("sandbox_session_id")

        # Same opt-in capture used by /chat (src/api/routes.py) — never
        # raises, true no-op unless Settings.capture_training_data is set.
        await capture_conversation(
            messages=result["messages"],
            enabled=_settings.capture_training_data,
            output_path=_settings.training_capture_path,
            pending_review_path=_settings.training_capture_pending_review_path,
        )

        return result["messages"][-1].content

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        # `new_message` is a livekit.agents.llm.ChatMessage, NOT a
        # langchain message — passing it directly into the graph fails
        # with "Unsupported message type" once it reaches the LLM client.
        # `.text_content` is the documented way to get its plain string
        # content, confirmed against the real installed SDK.
        user_text = new_message.text_content or ""
        response_text = await self._generate_reply_text_or_spoken_error(user_text)
        # `self.session` (not turn_ctx.session — ChatContext has no such
        # attribute) is how a running Agent accesses its AgentSession,
        # confirmed against the real livekit-agents API.
        await self.session.say(response_text)

    async def _generate_reply_text_or_spoken_error(self, user_text: str) -> str:
        """
        Wraps generate_reply_text so a backend failure (Redis down, E2B
        unreachable, Groq erroring, etc.) produces a spoken error instead
        of silence. Confirmed live: a Redis outage during a real call
        raised inside on_user_turn_completed, was caught and logged by
        LiveKit's own error handling, but nothing ever called
        `session.say(...)` on that turn — so the caller heard nothing and
        had no way to know the agent had failed versus just being slow.
        """
        try:
            return await self.generate_reply_text(user_text)
        except Exception as exc:  # noqa: BLE001 — this is the last-resort boundary
            logger.error("generate_reply_text failed for utterance %r: %s", user_text, exc)
            return (
                "Sorry, I hit a technical problem processing that. "
                "Could you try again in a moment?"
            )


def _build_sandbox_manager() -> SandboxManager:
    """
    Constructed once at module import, mirroring src/api/server.py's
    lifespan wiring but without needing an async context — Redis/E2B
    client construction itself is synchronous; only their I/O methods are
    async.
    """
    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(_settings.redis_url)
    registry = SessionRegistry(redis_client)
    policy = SandboxPolicy(
        default_timeout_seconds=_settings.sandbox_default_timeout_seconds,
        max_timeout_seconds=_settings.sandbox_max_timeout_seconds,
        max_output_chars=_settings.sandbox_max_output_chars,
        allow_shell=_settings.sandbox_allow_shell,
        allow_clone_repo=_settings.sandbox_allow_clone_repo,
    )
    return E2BSandboxManager(registry=registry, policy=policy, e2b_api_key=_settings.e2b_api_key)


_sandbox_manager = _build_sandbox_manager()

# Module-level AgentServer — discovered automatically by the LiveKit CLI
# (`python -m livekit.agents start src/voice/worker.py --dev` or the
# `console` subcommand for local testing without a real room). The name
# `app` is one of the conventional names the CLI's discovery mechanism
# looks for.
app = AgentServer(
    ws_url=_settings.livekit_url,
    api_key=_settings.livekit_api_key,
    api_secret=_settings.livekit_api_secret,
)


@app.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """LiveKit job entrypoint — one invocation per room/call."""
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    llm = build_chat_llm(_settings)
    if llm is None:
        raise RuntimeError(
            "Voice worker cannot start a session without a configured chat LLM. "
            "Set STREAMING_LLM_BASE_URL (and STREAMING_LLM_API_KEY if needed) in .env — "
            "see .env.example."
        )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(api_key=_settings.deepgram_api_key),
        tts=cartesia.TTS(api_key=_settings.cartesia_api_key),
        # Deliberately NOT passing llm= here. LiveKit's default text-input
        # handling (typing into the LiveKit Cloud console's chat box, for
        # example) calls AgentSession.generate_reply() directly against
        # whatever `llm` is configured on the session itself, bypassing
        # our on_user_turn_completed override entirely — which would mean
        # typed messages skip our LangGraph orchestrator (and its tools)
        # completely, and previously failed outright since no llm was set.
        # The custom text_input_cb below routes typed input through the
        # same graph as spoken input instead.
    )

    agent = NexusVoiceAgent(sandbox_manager=_sandbox_manager, thread_id=ctx.room.name, llm=llm)

    async def _text_input_cb(sess: AgentSession, ev) -> None:
        """Routes typed chat-box input through the same LangGraph path as
        spoken input, instead of LiveKit's default (which requires
        AgentSession's own `llm` and bypasses our orchestrator/tools)."""
        await sess.interrupt()
        reply_text = await agent._generate_reply_text_or_spoken_error(ev.text)
        await sess.say(reply_text)

    await session.start(
        agent=agent,
        room=ctx.room,
        # room_options (not the older, deprecated room_input_options /
        # room_output_options split) — confirmed against the real SDK,
        # which logs a deprecation warning for the old params.
        room_options=RoomOptions(text_input=TextInputOptions(text_input_cb=_text_input_cb)),
    )
    logger.info("Voice session started for room=%s", ctx.room.name)

