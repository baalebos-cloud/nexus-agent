from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.api.config import Settings
from src.api.llm import build_chat_llm


def test_build_chat_llm_returns_none_when_unconfigured():
    settings = Settings(streaming_llm_base_url=None)
    assert build_chat_llm(settings) is None


def test_build_chat_llm_returns_chat_openai_when_configured():
    settings = Settings(
        streaming_llm_base_url="https://api.groq.com/openai/v1",
        streaming_llm_api_key="fake-key",
        streaming_llm_model="llama-3.3-70b-versatile",
    )
    llm = build_chat_llm(settings)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "llama-3.3-70b-versatile"


def test_build_chat_llm_uses_placeholder_key_for_keyless_vllm_deployments():
    """Self-hosted vLLM often has no auth in front of it; ChatOpenAI still
    requires a non-empty api_key string, so a placeholder must be used
    rather than failing construction."""
    settings = Settings(
        streaming_llm_base_url="http://localhost:8000/v1",
        streaming_llm_api_key=None,
    )
    llm = build_chat_llm(settings)
    assert llm is not None
