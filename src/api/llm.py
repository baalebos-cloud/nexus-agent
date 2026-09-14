"""
Streaming LLM client factory.

Both `src/api/server.py` (for `/chat`) and `src/voice/worker.py` (for the
low-latency voice pipeline) need the same client: an OpenAI-compatible
chat model pointed at either Groq's hosted API or a self-hosted vLLM
server, per PROJECT_CONTEXT.md's Model Layer. `ChatOpenAI` works for both,
since Groq exposes an OpenAI-compatible endpoint and vLLM's `serve` command
does too — only `base_url` (and whether a key is needed) differs.

Centralizing this here means switching from Groq to self-hosted vLLM later
is purely a `.env` change, not a code change in either caller.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.api.config import Settings

logger = logging.getLogger(__name__)


def build_chat_llm(settings: Settings) -> BaseChatModel | None:
    """
    Build the streaming chat LLM client from Settings, or return None if
    it isn't configured yet.

    Returning None (rather than raising) lets callers degrade gracefully —
    server.py sets app.state.chat_llm = None and /chat responds 503 instead
    of failing app startup; worker.py should do the equivalent for the
    voice pipeline.
    """
    if not settings.streaming_llm_base_url:
        logger.warning(
            "STREAMING_LLM_BASE_URL not set — chat LLM unavailable. "
            "See .env.example for Groq/vLLM setup instructions."
        )
        return None

    # vLLM's OpenAI-compatible server typically doesn't require a real key;
    # ChatOpenAI still needs a non-empty string, so fall back to a
    # placeholder rather than failing construction over a self-hosted
    # deployment with no auth in front of it.
    api_key = settings.streaming_llm_api_key or "not-required"

    return ChatOpenAI(
        base_url=settings.streaming_llm_base_url,
        api_key=api_key,
        model=settings.streaming_llm_model,
        streaming=True,
    )
