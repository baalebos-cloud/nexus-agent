"""
QLoRA fine-tuning script for the Model Layer described in
PROJECT_CONTEXT.md: Qwen 2.5 72B / Llama 3.3 70B, trained via QLoRA for
strict function calling and structured JSON reasoning.

IMPORTANT — verification status: this script is correct against the
documented, current APIs of transformers/peft/trl/bitsandbytes (checked
against their published interfaces), but it has NOT been executed in this
build environment. A 70B-parameter QLoRA run needs a GPU with ~48GB+ VRAM
and the full torch/CUDA/bitsandbytes stack (torch alone is a 555MB wheel
plus several GB of CUDA dependencies) — installing that stack here risked
exhausting this sandbox's ~5GB of free disk entirely, so it was
deliberately not attempted. Run this on real GPU infrastructure, and treat
the data pipeline (src/finetune/data_schema.py, formatting.py,
dataset_prep.py) — which IS fully tested in this repo — as the verified
part of this subsystem.

Usage:
    python -m src.finetune.train_qlora \\
        --train-file data/prepared/train.jsonl \\
        --val-file data/prepared/val.jsonl \\
        --base-model Qwen/Qwen2.5-72B-Instruct \\
        --output-dir ./qlora-nexus-agent \\
        --num-epochs 3
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Nexus-Agent's model layer.")
    parser.add_argument("--train-file", required=True, help="Path to train.jsonl (from dataset_prep.py).")
    parser.add_argument("--val-file", required=True, help="Path to val.jsonl (from dataset_prep.py).")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="HF Hub model ID or local path. Llama-3.3-70B-Instruct is the alternate target "
        "per PROJECT_CONTEXT.md — remember to switch src/finetune/formatting.py's formatter "
        "(format_example_llama3 instead of format_example_qwen) in dataset_prep.py if so.",
    )
    parser.add_argument("--output-dir", required=True, help="Where to save LoRA adapter weights.")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # Imports deliberately deferred into main() rather than module level:
    # this lets `--help` and argument validation work even on a machine
    # without torch/CUDA installed, and makes the "this needs a GPU
    # machine" boundary explicit rather than an import-time crash with a
    # confusing traceback.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer

    logging.basicConfig(level=logging.INFO)

    logger.info("Loading tokenizer and base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit NF4 quantization — the "Q" in QLoRA. Loads a 70B-class model
    # into a fraction of its native fp16 memory footprint, making
    # fine-tuning feasible on a single high-memory GPU instead of a
    # multi-node cluster.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Standard attention + MLP projection targets for Qwen2/Llama-family
        # architectures — both use the same module naming convention.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset: train=%s val=%s", args.train_file, args.val_file)
    dataset = load_dataset(
        "json", data_files={"train": args.train_file, "validation": args.val_file}
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",  # matches dataset_prep.py's write_formatted_jsonl output
        packing=False,  # keep conversations un-packed so tool-call structure isn't split mid-sequence
        report_to="none",  # set to "wandb"/"tensorboard" as needed for your infra
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
