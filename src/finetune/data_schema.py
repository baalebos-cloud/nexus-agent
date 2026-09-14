"""
Data schemas for the fine-tuning pipeline.

Per PROJECT_CONTEXT.md's Model Layer, the target is a Qwen 2.5 72B /
Llama 3.3 70B model fine-tuned via QLoRA for strict function calling and
structured JSON reasoning. This module defines the training example
format — deliberately shaped to match what the live orchestrator already
produces (src/orchestrator/agent.py's SYSTEM_PROMPT, tool schemas from
sandbox_tools.py / osint_tools.py) — so real conversation transcripts
captured from the running system can become training data with minimal
transformation, rather than requiring a separate synthetic-data format.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A single function/tool invocation the assistant chose to make."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Parsed tool arguments (must be valid JSON-serializable)."
    )


class Turn(BaseModel):
    """
    One turn in a training conversation.

    Mirrors the shape LangChain/OpenAI-style chat messages already use in
    this codebase (see src/orchestrator/agent.py), so converting a real
    captured transcript into a Turn sequence is closer to a reshape than
    a reformat.
    """

    role: Role
    content: str | None = Field(
        default=None, description="Text content. None for pure tool-call turns."
    )
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = Field(
        default=None, description="Set on role=TOOL turns; matches the ToolCall.id it answers."
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> "Turn":
        if self.role == Role.TOOL and self.tool_call_id is None:
            raise ValueError("role=TOOL turns must set tool_call_id.")
        if self.role != Role.ASSISTANT and self.tool_calls:
            raise ValueError("Only role=ASSISTANT turns may set tool_calls.")
        if self.content is None and not self.tool_calls and self.role != Role.TOOL:
            raise ValueError("A turn must have content, tool_calls, or be a TOOL response.")
        return self


class TrainingExample(BaseModel):
    """
    One complete training example: a full conversation ending in the
    assistant's correct response (text and/or tool calls).

    `source` is required and deliberately not free-text — every example
    must be traceable to where it came from, which matters for both
    dataset auditing and for excluding accidentally-captured sensitive
    content (e.g. a real OSINT analysis transcript) from training data
    without a manual per-example review pass.
    """

    example_id: str
    turns: list[Turn] = Field(..., min_length=2)
    source: Literal["synthetic", "live_capture_sandbox", "live_capture_osint", "human_authored"]
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_conversation_shape(self) -> "TrainingExample":
        if self.turns[0].role != Role.SYSTEM and self.turns[0].role != Role.USER:
            raise ValueError("First turn must be role=SYSTEM or role=USER.")
        if self.turns[-1].role != Role.ASSISTANT:
            raise ValueError(
                "Last turn must be role=ASSISTANT — a training example must end with "
                "the response being trained on, not a dangling tool call or user turn."
            )
        # Never train on live OSINT capture without an explicit review tag —
        # this is a structural safeguard, not just documentation, mirroring
        # the AnalysisScope gate pattern used elsewhere in this project.
        if self.source == "live_capture_osint" and "reviewed" not in self.tags:
            raise ValueError(
                "source='live_capture_osint' examples must be tagged 'reviewed' before "
                "they can be used as training data — OSINT transcripts may contain "
                "sensitive identity-analysis content that needs human sign-off first."
            )
        return self
