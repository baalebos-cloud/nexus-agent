"""
Formats TrainingExample conversations into the target model's chat
template.

Qwen2.5 (the primary target per PROJECT_CONTEXT.md) uses ChatML framing
(`<|im_start|>{role}\\n{content}<|im_end|>\\n`) with tool calls embedded in
the assistant turn's content as `<tool_call>{json}</tool_call>` blocks,
and tool results fed back as a `tool` role turn wrapped in
`<tool_response>{json}</tool_response>` — this is Qwen2.5's actual,
documented function-calling convention (Hermes-style), not an invented
format, since training against a format the base model wasn't
pretrained/instruction-tuned to expect would actively hurt convergence.

Llama 3.3 uses a different template (`<|start_header_id|>` framing) — see
`format_llama3` below for the alternate path if that becomes the target
model instead.
"""

from __future__ import annotations

import json

from src.finetune.data_schema import Role, TrainingExample, Turn

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _format_turn_qwen(turn: Turn) -> str:
    if turn.role == Role.TOOL:
        body = f"<tool_response>\n{turn.content}\n</tool_response>"
        return f"{_IM_START}tool\n{body}{_IM_END}\n"

    if turn.role == Role.ASSISTANT and turn.tool_calls:
        parts = []
        if turn.content:
            parts.append(turn.content)
        for call in turn.tool_calls:
            call_json = json.dumps({"name": call.name, "arguments": call.arguments})
            parts.append(f"<tool_call>\n{call_json}\n</tool_call>")
        body = "\n".join(parts)
        return f"{_IM_START}assistant\n{body}{_IM_END}\n"

    return f"{_IM_START}{turn.role.value}\n{turn.content or ''}{_IM_END}\n"


def format_example_qwen(example: TrainingExample) -> str:
    """
    Produces the full ChatML text blob for one training example, suitable
    for a `text`-column SFT dataset (trl's SFTTrainer accepts pre-formatted
    text directly, avoiding the need to reimplement chat-template logic
    inside the training loop).
    """
    return "".join(_format_turn_qwen(turn) for turn in example.turns)


def format_example_llama3(example: TrainingExample) -> str:
    """
    Alternate formatter for Llama 3.3's header-based template, in case
    that becomes the fine-tuning target instead of Qwen2.5. Llama 3's
    tool-calling convention represents tool calls as a JSON object
    directly in the assistant message content (no special tags), per
    Meta's documented format.
    """
    parts: list[str] = []
    for turn in example.turns:
        if turn.role == Role.TOOL:
            content = f"<tool_response>{turn.content}</tool_response>"
            parts.append(f"<|start_header_id|>ipython<|end_header_id|>\n\n{content}<|eot_id|>")
            continue

        if turn.role == Role.ASSISTANT and turn.tool_calls:
            call = turn.tool_calls[0]  # Llama 3's format emits one call per turn
            content = json.dumps({"name": call.name, "parameters": call.arguments})
        else:
            content = turn.content or ""

        parts.append(f"<|start_header_id|>{turn.role.value}<|end_header_id|>\n\n{content}<|eot_id|>")

    return "".join(parts)
