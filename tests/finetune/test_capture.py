from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.finetune.capture import capture_conversation, promote_reviewed_capture
from src.finetune.data_schema import TrainingExample


def _tool_calling_conversation() -> list:
    return [
        SystemMessage(content="You are Nexus-Agent."),
        HumanMessage(content="Run print(1+1)"),
        AIMessage(
            content="",
            tool_calls=[{"name": "sandbox_run_code", "args": {"code": "print(1+1)"}, "id": "call_1"}],
        ),
        ToolMessage(content="2", tool_call_id="call_1"),
        AIMessage(content="The result is 2."),
    ]


def _osint_conversation() -> list:
    return [
        HumanMessage(content="Analyze this video for faces"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "osint_analyze_video_for_faces", "args": {"video_path": "x.mp4"}, "id": "call_1"}
            ],
        ),
        ToolMessage(content="No candidate matches found.", tool_call_id="call_1"),
        AIMessage(content="No matches found in the video."),
    ]


@pytest.mark.asyncio
async def test_capture_disabled_is_a_true_no_op():
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "captured.jsonl")
        pending_path = str(Path(tmp) / "pending.jsonl")

        await capture_conversation(
            _tool_calling_conversation(), enabled=False, output_path=out_path, pending_review_path=pending_path
        )

        assert not Path(out_path).exists()
        assert not Path(pending_path).exists()


@pytest.mark.asyncio
async def test_capture_writes_valid_training_example():
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "captured.jsonl")
        pending_path = str(Path(tmp) / "pending.jsonl")

        await capture_conversation(
            _tool_calling_conversation(), enabled=True, output_path=out_path, pending_review_path=pending_path
        )

        assert Path(out_path).exists()
        assert not Path(pending_path).exists()

        with open(out_path) as f:
            record = json.loads(f.readline())

        # Must round-trip through the real schema, not just be "some JSON".
        example = TrainingExample.model_validate(record)
        assert example.source == "live_capture_sandbox"
        assert example.turns[-1].role.value == "assistant"
        assert example.turns[-1].content == "The result is 2."


@pytest.mark.asyncio
async def test_capture_quarantines_osint_touched_conversations():
    """
    Structural safeguard, not a preference: a conversation that invoked an
    OSINT tool must NEVER be auto-written as a validated training example
    — data_schema.py's TrainingExample already refuses to construct
    source="live_capture_osint" without a "reviewed" tag, and no automated
    path is allowed to add that tag.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "captured.jsonl")
        pending_path = str(Path(tmp) / "pending.jsonl")

        await capture_conversation(
            _osint_conversation(), enabled=True, output_path=out_path, pending_review_path=pending_path
        )

        assert not Path(out_path).exists()
        assert Path(pending_path).exists()

        with open(pending_path) as f:
            record = json.loads(f.readline())
        assert record["intended_source"] == "live_capture_osint"
        assert "pending_id" in record
        # Confirm it's genuinely NOT a valid TrainingExample as stored —
        # it lacks example_id/source/tags, proving it was never
        # constructed as one.
        assert "source" not in record


@pytest.mark.asyncio
async def test_capture_skips_incomplete_conversations():
    """A conversation that doesn't end on a real assistant response (e.g.
    captured mid-tool-call) isn't a valid training shape — should be
    silently skipped, not raise or write malformed data."""
    incomplete = _tool_calling_conversation()[:-1]  # drop the final AIMessage

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "captured.jsonl")
        pending_path = str(Path(tmp) / "pending.jsonl")

        await capture_conversation(
            incomplete, enabled=True, output_path=out_path, pending_review_path=pending_path
        )

        assert not Path(out_path).exists()
        assert not Path(pending_path).exists()


@pytest.mark.asyncio
async def test_capture_never_raises_on_internal_failure():
    """Capture failures must never break the actual chat/voice response
    the caller is waiting on."""
    # Passing a non-list should be handled gracefully, not propagate.
    await capture_conversation(
        messages=None,  # type: ignore[arg-type]
        enabled=True,
        output_path="/nonexistent/deeply/nested/path.jsonl",
        pending_review_path="/nonexistent/pending.jsonl",
    )
    # No assertion needed beyond "this didn't raise."


@pytest.mark.asyncio
async def test_capture_appends_multiple_conversations():
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "captured.jsonl")
        pending_path = str(Path(tmp) / "pending.jsonl")

        await capture_conversation(
            _tool_calling_conversation(), enabled=True, output_path=out_path, pending_review_path=pending_path
        )
        await capture_conversation(
            [HumanMessage(content="hi"), AIMessage(content="hello")],
            enabled=True,
            output_path=out_path,
            pending_review_path=pending_path,
        )

        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 2


def test_promote_reviewed_capture_moves_approved_entries():
    with tempfile.TemporaryDirectory() as tmp:
        pending_path = str(Path(tmp) / "pending.jsonl")
        output_path = str(Path(tmp) / "captured.jsonl")

        import asyncio

        asyncio.run(
            capture_conversation(
                _osint_conversation(), enabled=True, output_path=output_path, pending_review_path=pending_path
            )
        )

        with open(pending_path) as f:
            pending_record = json.loads(f.readline())
        pending_id = pending_record["pending_id"]

        promoted_count = promote_reviewed_capture(
            pending_review_path=pending_path,
            approved_pending_ids={pending_id},
            output_path=output_path,
        )

        assert promoted_count == 1
        with open(output_path) as f:
            promoted_record = json.loads(f.readline())

        example = TrainingExample.model_validate(promoted_record)
        assert example.source == "live_capture_osint"
        assert "reviewed" in example.tags
        assert example.example_id == pending_id


def test_promote_reviewed_capture_skips_unapproved_entries():
    with tempfile.TemporaryDirectory() as tmp:
        pending_path = str(Path(tmp) / "pending.jsonl")
        output_path = str(Path(tmp) / "captured.jsonl")

        import asyncio

        asyncio.run(
            capture_conversation(
                _osint_conversation(), enabled=True, output_path=output_path, pending_review_path=pending_path
            )
        )

        promoted_count = promote_reviewed_capture(
            pending_review_path=pending_path,
            approved_pending_ids=set(),  # nothing approved
            output_path=output_path,
        )

        assert promoted_count == 0
        assert not Path(output_path).exists()
