from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.orchestrator.agent import build_agent_graph
from src.sandbox.models import SandboxBackend, SandboxSession, Runtime


class FakeLLM:
    """
    Minimal stand-in for a BaseChatModel: `bind_tools` is a no-op returning
    self, `ainvoke` returns a scripted AIMessage with no tool calls so the
    graph terminates after one agent-node pass. This proves the graph
    wiring (provision -> agent -> END) actually executes, without needing
    a real model API key.
    """

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="Hello from the agent.")


@pytest.mark.asyncio
async def test_agent_graph_runs_end_to_end_without_tool_calls():
    fake_manager = AsyncMock()
    fake_manager.get_session = AsyncMock(
        return_value=SandboxSession(
            backend=SandboxBackend.E2B,
            backend_sandbox_id="sbx-1",
            runtime=Runtime.PYTHON,
        )
    )
    fake_manager.create_session = AsyncMock(
        return_value=SandboxSession(
            session_id="sess-1",
            backend=SandboxBackend.E2B,
            backend_sandbox_id="sbx-1",
            runtime=Runtime.PYTHON,
        )
    )

    graph = build_agent_graph(llm=FakeLLM(), sandbox_manager=fake_manager)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="hi")],
            "thread_id": "thread-1",
            "sandbox_session_id": None,
            "osint_scope": None,
            "is_complete": False,
            "scratch": {},
        }
    )

    assert result["sandbox_session_id"] == "sess-1"
    assert result["messages"][-1].content == "Hello from the agent."
    fake_manager.create_session.assert_awaited_once()
