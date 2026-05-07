#!/usr/bin/env python3
"""MCQA visual grounding pretraining on Qwen2.5-VL.

Stage-0: Train the model to answer multiple-choice questions that require
looking at the image to distinguish correct answers from hard negatives.

Usage:
  accelerate launch --config_file accelerate_config.yaml train_mcqa.py \
    --model_id /path/to/Qwen2.5-VL-7B-Instruct \
    --mcqa_data /path/to/covt_part6/mcqa.jsonl \
    --eval_data /path/to/covt_part6/mcqa_val_50.jsonl \
    --image_folder /path/to/covt_part6/images \
    --output_dir ./out_mcqa_pretrain
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from dataclasses import dataclass, field
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.mcqa_dataset import MCQADataset, MCQACollator


@dataclass
class ModelArgs:
    model_id: str = field(metadata={"help": "Qwen2.5-VL model path."})


@dataclass
class DataArgs:
    mcqa_data: str = field(metadata={"help": "Path to mcqa.jsonl."})
    image_folder: str = field(metadata={"help": "Folder containing images."})
    eval_data: str | None = field(default=None, metadata={"help": "Optional validation JSONL."})
    max_seq_length: int = field(default=2048)
    max_pixels: int = field(default=401408)


@dataclass
class MCQATrainingArgs(TrainingArguments):
    freeze_vision: bool = field(default=True)
    disable_flash_attn2: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)


def _freeze_vision(model: torch.nn.Module) -> None:
    sub = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if sub is not None:
        for p in sub.parameters():
            p.requires_grad = False


def main() -> None:
    parser = HfArgumentParser((ModelArgs, DataArgs, MCQATrainingArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    proc = AutoProcessor.from_pretrained(
        model_args.model_id, trust_remote_code=True,
        max_pixels=data_args.max_pixels,
    )
    tok = proc.tokenizer

    attn = "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, torch_dtype=compute_dtype,
            attn_implementation=attn, trust_remote_code=True,
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, torch_dtype=compute_dtype,
            attn_implementation="sdpa", trust_remote_code=True,
        )

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if training_args.freeze_vision:
        _freeze_vision(model)

    train_ds = MCQADataset(
        data_path=data_args.mcqa_data,
        image_folder=data_args.image_folder,
        processor=proc,
        tokenizer=tok,
        max_seq_length=data_args.max_seq_length,
    )
    eval_ds = None
    if data_args.eval_data and os.path.isfile(data_args.eval_data):
        eval_ds = MCQADataset(
            data_path=data_args.eval_data,
            image_folder=data_args.image_folder,
            processor=proc,
            tokenizer=tok,
            max_seq_length=data_args.max_seq_length,
        )
    collator = MCQACollator(pad_token_id=tok.pad_token_id or 0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=proc,
    )
    trainer.train()

    model.config.use_cache = True
    if trainer.is_world_process_zero():
        trainer.save_model()


if __name__ == "__main__":
    main()
