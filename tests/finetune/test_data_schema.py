from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.finetune.data_schema import Role, ToolCall, TrainingExample, Turn


def _basic_turns() -> list[Turn]:
    return [
        Turn(role=Role.SYSTEM, content="You are Nexus-Agent."),
        Turn(role=Role.USER, content="Hello"),
        Turn(role=Role.ASSISTANT, content="Hi there!"),
    ]


def test_valid_example_constructs():
    example = TrainingExample(example_id="ex-1", turns=_basic_turns(), source="synthetic")
    assert example.example_id == "ex-1"


def test_example_must_end_with_assistant_turn():
    turns = _basic_turns()[:-1]  # drop the trailing assistant turn -> ends on USER
    with pytest.raises(ValidationError, match="Last turn must be role=ASSISTANT"):
        TrainingExample(example_id="ex-2", turns=turns, source="synthetic")


def test_tool_turn_requires_tool_call_id():
    with pytest.raises(ValidationError, match="tool_call_id"):
        Turn(role=Role.TOOL, content="result")


def test_only_assistant_may_have_tool_calls():
    with pytest.raises(ValidationError, match="Only role=ASSISTANT"):
        Turn(
            role=Role.USER,
            content="hi",
            tool_calls=[ToolCall(id="c1", name="foo", arguments={})],
        )


def test_osint_capture_requires_reviewed_tag():
    """
    Structural safeguard: OSINT transcripts may contain sensitive
    identity-analysis content and must not become training data without
    explicit human review — mirrors the AnalysisScope gate pattern used
    for the OSINT pipeline itself.
    """
    with pytest.raises(ValidationError, match="reviewed"):
        TrainingExample(
            example_id="ex-osint",
            turns=_basic_turns(),
            source="live_capture_osint",
            tags=[],
        )


def test_osint_capture_with_reviewed_tag_succeeds():
    example = TrainingExample(
        example_id="ex-osint-ok",
        turns=_basic_turns(),
        source="live_capture_osint",
        tags=["reviewed"],
    )
    assert "reviewed" in example.tags


def test_turn_requires_content_or_tool_calls():
    with pytest.raises(ValidationError, match="content, tool_calls"):
        Turn(role=Role.ASSISTANT, content=None, tool_calls=[])
