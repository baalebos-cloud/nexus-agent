from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.finetune.data_schema import Role, TrainingExample, Turn
from src.finetune.dataset_prep import (
    DatasetValidationError,
    load_examples,
    prepare_dataset,
    split_train_val,
    write_formatted_jsonl,
)


def _write_raw_jsonl(path: str, n_examples: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_examples):
            example = TrainingExample(
                example_id=f"ex-{i}",
                source="synthetic",
                turns=[
                    Turn(role=Role.USER, content=f"question {i}"),
                    Turn(role=Role.ASSISTANT, content=f"answer {i}"),
                ],
            )
            f.write(example.model_dump_json() + "\n")


def test_load_examples_reads_valid_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "raw.jsonl")
        _write_raw_jsonl(path, 5)
        examples = load_examples(path)
        assert len(examples) == 5
        assert examples[0].example_id == "ex-0"


def test_load_examples_strict_raises_on_invalid_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "raw.jsonl")
        with open(path, "w") as f:
            f.write('{"not": "a valid example"}\n')
        with pytest.raises(DatasetValidationError):
            load_examples(path, strict=True)


def test_load_examples_non_strict_skips_invalid_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "raw.jsonl")
        with open(path, "w") as f:
            f.write('{"not": "a valid example"}\n')
            valid = TrainingExample(
                example_id="ex-ok",
                source="synthetic",
                turns=[Turn(role=Role.USER, content="hi"), Turn(role=Role.ASSISTANT, content="hello")],
            )
            f.write(valid.model_dump_json() + "\n")
        examples = load_examples(path, strict=False)
        assert len(examples) == 1
        assert examples[0].example_id == "ex-ok"


def test_split_train_val_is_deterministic_with_fixed_seed():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "raw.jsonl")
        _write_raw_jsonl(path, 20)
        examples = load_examples(path)

    train1, val1 = split_train_val(examples, val_fraction=0.2, seed=42)
    train2, val2 = split_train_val(examples, val_fraction=0.2, seed=42)

    assert [e.example_id for e in train1] == [e.example_id for e in train2]
    assert [e.example_id for e in val1] == [e.example_id for e in val2]
    assert len(val1) == 4  # 20 * 0.2
    assert len(train1) == 16


def test_split_train_val_no_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "raw.jsonl")
        _write_raw_jsonl(path, 10)
        examples = load_examples(path)

    train, val = split_train_val(examples, val_fraction=0.3, seed=1)
    train_ids = {e.example_id for e in train}
    val_ids = {e.example_id for e in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {e.example_id for e in examples}


def test_split_train_val_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_train_val([], val_fraction=1.5)


def test_write_formatted_jsonl_produces_text_field():
    example = TrainingExample(
        example_id="ex-1",
        source="synthetic",
        turns=[Turn(role=Role.USER, content="hi"), Turn(role=Role.ASSISTANT, content="hello")],
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "out.jsonl")
        write_formatted_jsonl([example], out_path)

        with open(out_path) as f:
            line = json.loads(f.readline())
        assert line["example_id"] == "ex-1"
        assert "<|im_start|>user" in line["text"]


def test_prepare_dataset_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = str(Path(tmp) / "raw.jsonl")
        output_dir = str(Path(tmp) / "prepared")
        _write_raw_jsonl(raw_path, 10)

        counts = prepare_dataset(raw_path, output_dir, val_fraction=0.2, seed=7)

        assert counts == {"total": 10, "train": 8, "val": 2}
        assert (Path(output_dir) / "train.jsonl").exists()
        assert (Path(output_dir) / "val.jsonl").exists()

        with open(Path(output_dir) / "train.jsonl") as f:
            train_lines = f.readlines()
        assert len(train_lines) == 8


def test_prepare_dataset_raises_on_empty_input():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = str(Path(tmp) / "empty.jsonl")
        Path(raw_path).touch()
        with pytest.raises(DatasetValidationError):
            prepare_dataset(raw_path, str(Path(tmp) / "out"))
