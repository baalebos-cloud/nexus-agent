"""
Live-usage capture: turns real /chat and voice conversation transcripts
into fine-tuning training data automatically.

This is the bridge from "the agent is being used" to
"src/finetune/dataset_prep.py has something to train on" — without it,
building a training set means hand-authoring synthetic examples forever.

Structural safeguard, not just documentation: any conversation where an
OSINT tool was invoked is NEVER auto-written as a validated
TrainingExample. data_schema.py's TrainingExample already refuses to
construct a source="live_capture_osint" example without a "reviewed" tag
— this module respects that by routing OSINT-touched conversations to a
separate, unvalidated quarantine file instead of attempting (and having
to catch the failure of) direct construction. A human reviews that file
and promotes entries deliberately; nothing here can silently leak
identity-analysis transcripts into a training set.

Usage (see also src/api/routes.py and src/voice/worker.py for wiring):

    from src.finetune.capture import capture_conversation

    await capture_conversation(
        messages=result["messages"],
        enabled=settings.capture_training_data,
        output_path=settings.training_capture_path,
        pending_review_path=settings.training_capture_pending_review_path,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.finetune.data_schema import Role, ToolCall, TrainingExample, Turn

logger = logging.getLogger(__name__)

# Tool names that indicate an OSINT identity-analysis capability was used
# in this conversation. Kept as an explicit allowlist here (rather than
# importing from osint_tools.py) so this module has no import-time
# dependency on the OSINT subsystem — capture should degrade gracefully
# even if that module changes shape.
_OSINT_TOOL_NAMES = frozenset({"osint_analyze_video_for_faces"})

# A single process-wide lock serializes appends to the capture files —
# multiple concurrent /chat requests must not interleave partial JSON
# lines into each other.
_write_lock = asyncio.Lock()


def _convert_messages_to_turns(messages: list[BaseMessage]) -> list[Turn]:
    """
    Converts a LangGraph/LangChain message list into our Turn schema.

    Confirmed against real langchain_core message shapes: AIMessage.tool_calls
    is a list of dicts with 'name'/'args'/'id' keys (not a custom object),
    and ToolMessage carries .tool_call_id directly.
    """
    turns: list[Turn] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            turns.append(Turn(role=Role.SYSTEM, content=str(msg.content)))
        elif isinstance(msg, HumanMessage):
            turns.append(Turn(role=Role.USER, content=str(msg.content)))
        elif isinstance(msg, ToolMessage):
            turns.append(
                Turn(role=Role.TOOL, content=str(msg.content), tool_call_id=msg.tool_call_id)
            )
        elif isinstance(msg, AIMessage):
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("args", {}))
                for tc in (msg.tool_calls or [])
            ]
            content = str(msg.content) if msg.content else None
            turns.append(Turn(role=Role.ASSISTANT, content=content, tool_calls=tool_calls))
        else:
            logger.warning("Skipping unrecognized message type during capture: %s", type(msg))
    return turns


def _conversation_used_osint_tools(turns: list[Turn]) -> bool:
    return any(
        tc.name in _OSINT_TOOL_NAMES
        for turn in turns
        for tc in turn.tool_calls
    )


async def _append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    async with _write_lock:
        # Plain blocking file I/O under the lock is fine here — capture
        # writes are small and infrequent relative to LLM/tool latency,
        # so this isn't a meaningful bottleneck. Using asyncio.to_thread
        # would add complexity for no real benefit at this volume.
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


async def capture_conversation(
    messages: list[BaseMessage],
    enabled: bool,
    output_path: str,
    pending_review_path: str,
) -> None:
    """
    Captures one completed conversation as training data, or quarantines
    it for human review if OSINT tools were used.

    Deliberately never raises — a capture failure must never break the
    actual chat/voice response the user is waiting on. All failures are
    logged and swallowed.

    `enabled=False` (the default via Settings.capture_training_data) is a
    true no-op, not just an early return with side effects still queued —
    capturing conversation transcripts has real privacy implications, so
    this must be explicit opt-in, not opt-out.
    """
    if not enabled:
        return

    try:
        turns = _convert_messages_to_turns(messages)
        if not turns or turns[-1].role != Role.ASSISTANT:
            # Conversation didn't end on a real assistant response (e.g.
            # captured mid-tool-call-loop) — not a valid training example
            # shape per TrainingExample's own validation, so skip rather
            # than log a spurious warning for normal in-progress state.
            return

        if _conversation_used_osint_tools(turns):
            # Quarantined as a raw, UNVALIDATED dict — never constructed
            # as a TrainingExample, since that constructor would reject
            # source="live_capture_osint" without a "reviewed" tag no
            # automated path is allowed to add. A human reviews this file
            # and promotes entries deliberately (see promote_reviewed_capture).
            record = {
                "pending_id": str(uuid.uuid4()),
                "turns": [t.model_dump(mode="json") for t in turns],
                "intended_source": "live_capture_osint",
            }
            await _append_jsonl(pending_review_path, record)
            logger.info("Conversation quarantined for OSINT review: %s", record["pending_id"])
            return

        example = TrainingExample(
            example_id=str(uuid.uuid4()),
            turns=turns,
            source="live_capture_sandbox",
        )
        await _append_jsonl(output_path, json.loads(example.model_dump_json()))
        logger.info("Captured live conversation as training example: %s", example.example_id)

    except Exception as exc:  # noqa: BLE001 — capture must never break the caller
        logger.error("Training data capture failed (non-fatal): %s", exc)


def promote_reviewed_capture(
    pending_review_path: str,
    approved_pending_ids: set[str],
    output_path: str,
) -> int:
    """
    Human-in-the-loop promotion step: reads the quarantine file, and for
    each entry whose pending_id is in `approved_pending_ids`, constructs a
    real TrainingExample tagged "reviewed" (satisfying the schema's
    safeguard) and appends it to the normal training-data output file.

    This is the ONLY code path that can turn an OSINT-touched capture into
    actual training data — and it requires a human to have already
    decided which pending_ids are approved (e.g. via manual review of the
    quarantine file's content), not an automated heuristic.
    """
    promoted = 0
    with open(pending_review_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["pending_id"] not in approved_pending_ids:
                continue

            turns = [Turn.model_validate(t) for t in record["turns"]]
            example = TrainingExample(
                example_id=record["pending_id"],
                turns=turns,
                source="live_capture_osint",
                tags=["reviewed"],
            )
            asyncio.run(_append_jsonl(output_path, json.loads(example.model_dump_json())))
            promoted += 1

    return promoted
