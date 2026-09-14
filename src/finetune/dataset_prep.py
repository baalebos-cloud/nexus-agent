"""
Dataset preparation for QLoRA fine-tuning.

Pipeline: raw JSONL (one TrainingExample per line) -> schema validation ->
train/val split -> formatted-text JSONL, ready for trl's SFTTrainer
(which accepts a "text" column directly).

Deliberately a separate, inspectable step from training itself — you
should be able to look at the exact formatted text a model will train on
before spending GPU-hours on it.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from pydantic import ValidationError

from src.finetune.data_schema import TrainingExample
from src.finetune.formatting import format_example_qwen

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when raw examples fail schema validation and strict=True."""


def load_examples(path: str, strict: bool = True) -> list[TrainingExample]:
    """
    Load and validate TrainingExamples from a JSONL file (one JSON object
    per line).

    strict=True (default) raises on the first invalid example — appropriate
    for CI/pre-training validation. strict=False logs and skips invalid
    lines instead, for exploratory work with a messier raw dataset.
    """
    examples: list[TrainingExample] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(TrainingExample.model_validate(data))
            except (json.JSONDecodeError, ValidationError) as exc:
                if strict:
                    raise DatasetValidationError(
                        f"{path}:{line_num}: invalid training example: {exc}"
                    ) from exc
                logger.warning("Skipping invalid example at %s:%d: %s", path, line_num, exc)
    return examples


def split_train_val(
    examples: list[TrainingExample], val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Deterministic shuffle-then-split. Fixed seed by default so dataset
    versions are reproducible across runs — an accidental reshuffle
    between training runs makes eval metrics incomparable.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive).")

    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]
    return train_set, val_set


def write_formatted_jsonl(examples: list[TrainingExample], output_path: str) -> None:
    """
    Writes one JSON object per line, each with a single "text" field
    containing the fully formatted ChatML conversation — the shape
    trl.SFTTrainer expects when given a dataset with a text column
    directly, rather than raw messages it would need to template itself.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            text = format_example_qwen(example)
            f.write(json.dumps({"text": text, "example_id": example.example_id}) + "\n")


def prepare_dataset(
    raw_path: str,
    output_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    strict: bool = True,
) -> dict[str, int]:
    """
    End-to-end: load raw examples, split, write train.jsonl + val.jsonl
    into output_dir. Returns example counts for logging/reporting.
    """
    examples = load_examples(raw_path, strict=strict)
    if not examples:
        raise DatasetValidationError(f"No valid training examples found in {raw_path!r}.")

    train_set, val_set = split_train_val(examples, val_fraction=val_fraction, seed=seed)

    write_formatted_jsonl(train_set, str(Path(output_dir) / "train.jsonl"))
    write_formatted_jsonl(val_set, str(Path(output_dir) / "val.jsonl"))

    counts = {"total": len(examples), "train": len(train_set), "val": len(val_set)}
    logger.info("Dataset prepared: %s", counts)
    return counts
