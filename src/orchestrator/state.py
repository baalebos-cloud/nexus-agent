"""
LangGraph agent state schema.

This is the single shape carried through the state graph in agent.py.
Sandbox session_id and OSINT scope live here, per the session-lifecycle
design from the Sandboxed Execution Engine build: a long-horizon task
reuses the same sandbox session across tool calls instead of
re-provisioning, and an OSINT scope (if any) is attached once per task
rather than re-derived per tool call.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

from src.osint.models import AnalysisScope


class AgentState(TypedDict, total=False):
    """
    LangGraph state dict.

    `messages` uses the standard add-messages reducer pattern (append, not
    overwrite) so tool-call/tool-result turns accumulate correctly across
    graph steps.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    thread_id: str

    # Sandbox — reused across tool calls within a thread (see
    # src/orchestrator/sandbox_tools.py:ensure_session_for_thread)
    sandbox_session_id: str | None

    # OSINT — orchestrator-issued, never LLM-settable (see
    # src/orchestrator/osint_tools.py)
    osint_scope: AnalysisScope | None

    # Set by the router node to short-circuit further tool calls when the
    # graph decides the task is complete.
    is_complete: bool

    # Free-form scratch space for node-to-node handoff that doesn't belong
    # in messages (e.g. last tool error, for retry/backoff logic).
    scratch: dict[str, Any]
