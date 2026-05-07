#!/usr/bin/env python3
"""PinReason v2 — two-stage latent visual reasoning SFT on Qwen2.5-VL.

Usage (8×A100):
  accelerate launch --config_file accelerate_config.yaml train.py \
    --model_id /path/to/Qwen2.5-VL-7B-Instruct \
    --train_data /path/to/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl \
    --eval_data  /path/to/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl \
    --image_folder /path/to/Visual-CoT/cot_extracted/cot_image_data/gqa \
    --output_dir ./out_v2_qwen_typed
"""

from __future__ import annotations

import os
import sys

os.environ.pop("TRANSFORMERS_CACHE", None)

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    HfArgumentParser,
    set_seed,
)

from src.constants import (
    BBOX_LVR_START,
    FULL_LVR_START,
    LVR,
    PATCH_LVR_START,
    PLAN_LVR_START,
    SPECIAL_TOKENS,
)
from src.dataset import GqaCotCollator, GqaCotDataset
from src.forward import install_pinreason_v2_patches
from src.params import DataArguments, ModelArguments, PinReasonV2TrainingArguments
from src.trainer import PinReasonV2Trainer


def _freeze_vision(model: torch.nn.Module) -> None:
    sub = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if sub is not None:
        for p in sub.parameters():
            p.requires_grad = False


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, PinReasonV2TrainingArguments))
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
    for t in SPECIAL_TOKENS:
        tok.add_tokens(t, special_tokens=True)

    install_pinreason_v2_patches(tokenizer=tok)

    cfg = AutoConfig.from_pretrained(model_args.model_id, trust_remote_code=True)
    attn = "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, config=cfg, torch_dtype=compute_dtype,
            attn_implementation=attn, trust_remote_code=True,
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, config=cfg, torch_dtype=compute_dtype,
            attn_implementation="sdpa", trust_remote_code=True,
        )
    model.resize_token_embeddings(len(tok))

    model.config.lvr_id = int(tok.convert_tokens_to_ids(LVR))
    model.config.plan_lvr_start_id = int(tok.convert_tokens_to_ids(PLAN_LVR_START))
    model.config.bbox_lvr_start_id = int(tok.convert_tokens_to_ids(BBOX_LVR_START))
    model.config.patch_lvr_start_id = int(tok.convert_tokens_to_ids(PATCH_LVR_START))
    model.config.full_lvr_start_id = int(tok.convert_tokens_to_ids(FULL_LVR_START))
    model.config.patch_cos_threshold = float(data_args.patch_cos_threshold)
    model.config.min_patches = int(data_args.min_patches)
    model.config.use_cache = False

    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        v = getattr(tok, attr, None)
        if v is not None:
            setattr(model.config, attr, v)
            gen = getattr(model, "generation_config", None)
            if gen is not None:
                setattr(gen, attr, v)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if training_args.freeze_vision:
        _freeze_vision(model)

    ds_kwargs = dict(
        image_folder=data_args.image_folder,
        processor=proc,
        tokenizer=tok,
        max_seq_length=data_args.max_seq_length,
        max_plan_tokens=data_args.max_plan_tokens,
        n_patch_tokens=data_args.n_patch_tokens,
        max_full_answer_tokens=data_args.max_full_answer_tokens,
        patch_fraction=data_args.patch_fraction,
        min_patches=data_args.min_patches,
    )
    train_ds = GqaCotDataset(data_path=data_args.train_data, **ds_kwargs)
    eval_ds = None
    if data_args.eval_data and os.path.isfile(data_args.eval_data):
        eval_ds = GqaCotDataset(data_path=data_args.eval_data, **ds_kwargs)

    collator = GqaCotCollator(pad_token_id=tok.pad_token_id or 0)

    if training_args.custom_warmup_ratio > 0 and training_args.warmup_steps == 0:
        import math
        n_samples = len(train_ds)
        effective_bs = (
            training_args.per_device_train_batch_size
            * max(1, torch.cuda.device_count())
            * training_args.gradient_accumulation_steps
        )
        steps_per_epoch = math.ceil(n_samples / effective_bs)
        total_steps = steps_per_epoch * int(training_args.num_train_epochs)
        training_args.warmup_steps = int(total_steps * training_args.custom_warmup_ratio)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[PinReason] {n_samples} samples, effective_bs={effective_bs}, "
                  f"total_steps={total_steps}, warmup_steps={training_args.warmup_steps}")

    trainer = PinReasonV2Trainer(
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
        trainer.save_model(os.path.join(training_args.output_dir, "best"))


if __name__ == "__main__":
    main()
