from __future__ import annotations

import json

from src.finetune.data_schema import Role, ToolCall, TrainingExample, Turn
from src.finetune.formatting import format_example_llama3, format_example_qwen


def _tool_calling_example() -> TrainingExample:
    return TrainingExample(
        example_id="ex-1",
        source="synthetic",
        turns=[
            Turn(role=Role.SYSTEM, content="You are Nexus-Agent."),
            Turn(role=Role.USER, content="Run print(1+1) in the sandbox."),
            Turn(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(id="call_1", name="sandbox_run_code", arguments={"code": "print(1+1)"})
                ],
            ),
            Turn(role=Role.TOOL, content="2", tool_call_id="call_1"),
            Turn(role=Role.ASSISTANT, content="The result is 2."),
        ],
    )


def test_qwen_format_uses_chatml_tags():
    text = format_example_qwen(_tool_calling_example())
    assert "<|im_start|>system" in text
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text
    assert "<|im_start|>tool" in text
    assert text.count("<|im_end|>") == 5  # one per turn


def test_qwen_format_embeds_tool_call_as_valid_json():
    text = format_example_qwen(_tool_calling_example())
    assert "<tool_call>" in text
    start = text.index("<tool_call>\n") + len("<tool_call>\n")
    end = text.index("\n</tool_call>")
    call_json = json.loads(text[start:end])
    assert call_json == {"name": "sandbox_run_code", "arguments": {"code": "print(1+1)"}}


def test_qwen_format_wraps_tool_response():
    text = format_example_qwen(_tool_calling_example())
    assert "<tool_response>\n2\n</tool_response>" in text


def test_qwen_format_simple_conversation_no_tools():
    example = TrainingExample(
        example_id="ex-simple",
        source="synthetic",
        turns=[
            Turn(role=Role.USER, content="Hello"),
            Turn(role=Role.ASSISTANT, content="Hi!"),
        ],
    )
    text = format_example_qwen(example)
    assert text == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n"


def test_llama3_format_uses_header_tags():
    text = format_example_llama3(_tool_calling_example())
    assert "<|start_header_id|>system<|end_header_id|>" in text
    assert "<|start_header_id|>ipython<|end_header_id|>" in text  # tool responses
    assert text.count("<|eot_id|>") == 5


def test_llama3_format_embeds_tool_call_as_json_no_tags():
    text = format_example_llama3(_tool_calling_example())
    # Llama 3's convention: raw JSON in content, no <tool_call> wrapper tags
    assert '"name": "sandbox_run_code"' in text
    assert "<tool_call>" not in text
