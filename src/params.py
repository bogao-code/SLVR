"""HF dataclass arguments for PinReason v2 (Qwen2.5-VL)."""

from __future__ import annotations
from dataclasses import dataclass, field
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_id: str = field(metadata={"help": "Qwen2.5-VL HF id or local path."})


@dataclass
class DataArguments:
    train_data: str = field(metadata={"help": "Path to training jsonl."})
    image_folder: str = field(metadata={"help": "Folder containing images."})
    eval_data: str = field(default="", metadata={"help": "Path to val jsonl."})
    max_seq_length: int = field(default=4096)
    max_pixels: int = field(default=401408)
    # Plan block: K1 = len(cot), capped at max_plan_tokens
    max_plan_tokens: int = field(default=128)
    # Optional cap for dynamic patch slots. 0 means no cap.
    n_patch_tokens: int = field(default=0)
    # Full-answer block: K4 = len(full_answer), capped at max_full_answer_tokens
    max_full_answer_tokens: int = field(default=48)
    # Patch selection
    patch_cos_threshold: float = field(default=0.5)
    patch_fraction: float = field(default=0.5)
    min_patches: int = field(default=5)


@dataclass
class PinReasonV2TrainingArguments(HFTrainingArguments):
    loss_lvr_fct: str = field(default="mse")
    # Per-step MSE loss weights
    # Step 1 (action plan): low — provides context, not critical precision
    loss_plan_lambda: float = field(default=0.1)
    # Step 2 (bounding box): high — precise localization matters
    loss_bbox_lambda: float = field(default=1.0)
    # Step 3 (patch embeddings): high — visual evidence for answer
    loss_patch_lambda: float = field(default=1.0)
    # Step 4 (full answer): medium
    loss_answer_lambda: float = field(default=0.5)
    # Extra CE on the transition from final latent block to <answer>
    loss_answer_open_lambda: float = field(default=2.0)
    # Extra CE on the final short answer tokens inside <answer>...</answer>
    loss_short_answer_lambda: float = field(default=3.0)
    freeze_vision: bool = field(default=True)
    disable_flash_attn2: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    log_loss_components: bool = field(default=True)
    custom_warmup_ratio: float = field(default=0.03)
