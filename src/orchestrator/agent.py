"""
LangGraph orchestrator — the agent's central state machine.

Graph shape:

    provision -> agent -> [conditional: tools | END]
                   ^          |
                   +----------+

`provision` ensures a sandbox session exists for this thread before the LLM
gets tool access (so `sandbox_run_code` etc. always have a live session_id
to target). `agent` calls the LLM with tools bound. If the LLM emits tool
calls, execution routes to `tools` (a prebuilt `ToolNode`) and loops back to
`agent`; otherwise the graph ends.

OSINT tools are only bound when the caller supplies an `AnalysisScope` at
graph-build time (see `build_agent_graph`'s `osint_scope` param) — if no
scope is supplied, the agent simply has no OSINT capability for that
invocation, which is the safe default per the access-control design.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.orchestrator.osint_tools import build_osint_tools
from src.orchestrator.sandbox_tools import build_sandbox_tools, ensure_session_for_thread
from src.orchestrator.state import AgentState
from src.osint.face_recognizer import FaceRecognizer
from src.osint.models import AnalysisScope
from src.osint.video_processor import VideoProcessor
from src.sandbox.manager import SandboxManager
from src.sandbox.models import Runtime

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Nexus-Agent, an autonomous research and analysis assistant. "
    "You have access to a sandboxed code execution environment and, when "
    "explicitly authorized for a task, video identity-analysis tools. "
    "Identity-match results are probabilistic — always relay the confidence "
    "framing given to you verbatim rather than restating a match as a "
    "certain fact."
)


def build_agent_graph(
    llm: BaseChatModel,
    sandbox_manager: SandboxManager,
    video_processor: VideoProcessor | None = None,
    face_recognizer: FaceRecognizer | None = None,
    osint_scope: AnalysisScope | None = None,
):
    """
    Construct the compiled LangGraph agent.

    `osint_scope` should come from the orchestrator's own request-handling
    layer (src/api/server.py routes), never from the LLM or from
    user-supplied free text — see osint_tools.py docstring.
    """

    async def provision_node(state: AgentState) -> AgentState:
        session_id = await ensure_session_for_thread(
            manager=sandbox_manager,
            thread_id=state["thread_id"],
            runtime=Runtime.PYTHON,
            existing_session_id=state.get("sandbox_session_id"),
        )
        return {"sandbox_session_id": session_id}

    def _bound_tools(state: AgentState) -> list:
        tools = list(
            build_sandbox_tools(manager=sandbox_manager, session_id=state["sandbox_session_id"])
        )
        active_scope = state.get("osint_scope") or osint_scope
        if active_scope is not None and video_processor is not None and face_recognizer is not None:
            tools.extend(
                build_osint_tools(
                    video_processor=video_processor,
                    face_recognizer=face_recognizer,
                    scope=active_scope,
                )
            )
        return tools

    async def agent_node(state: AgentState) -> AgentState:
        tools = _bound_tools(state)
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        response: AIMessage = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            return "tools"
        return END

    async def tools_node(state: AgentState) -> AgentState:
        # Rebuilt per-call rather than cached on the graph, since tool
        # closures capture `sandbox_session_id` / `osint_scope`, both of
        # which can legitimately change between turns of the same thread.
        tools = _bound_tools(state)
        node = ToolNode(tools)
        return await node.ainvoke(state)

    graph = StateGraph(AgentState)
    graph.add_node("provision", provision_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("provision")
    graph.add_edge("provision", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
