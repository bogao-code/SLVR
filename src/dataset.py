"""GQA-CoT dataset for PinReason v2 on Qwen2.5-VL — 4-step latent reasoning.

Sequence layout (assistant turn):
    <plan_lvr_start>  <lvr>×K1 <plan_lvr_end>
    <bbox_lvr_start>  <lvr>×3  <bbox_lvr_end>
    <patch_lvr_start> <lvr>×K3 <patch_lvr_end>
    <full_lvr_start>  <lvr>×K4 <full_lvr_end>
    <answer> short_answer </answer>
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.bbox_utils import bbox_to_token_idxs_lvr_packed
from src.constants import (
    ANSWER_CLOSE, ANSWER_OPEN, IGNORE_INDEX,
    BBOX_LVR_END, BBOX_LVR_START,
    FULL_LVR_END, FULL_LVR_START,
    LVR,
    PATCH_LVR_END, PATCH_LVR_START,
    PLAN_LVR_END, PLAN_LVR_START,
)


def _normalize_bbox(bbox, w: int, h: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if max(x0, y0, x1, y1) > 1.0 + 1e-6:
        x0, y0, x1, y1 = x0 / max(w, 1), y0 / max(h, 1), x1 / max(w, 1), y1 / max(h, 1)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(x0 + 1e-3, min(1.0, x1))
    y1 = max(y0 + 1e-3, min(1.0, y1))
    return (x0, y0, x1, y1)


def _xyxy_to_cxcywh(x0: float, y0: float, x1: float, y1: float
                    ) -> Tuple[float, float, float, float]:
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w  = x1 - x0
    h  = y1 - y0
    return cx, cy, w, h


def _build_assistant_text(n_plan: int, n_patch: int, n_answer: int,
                           short_answer: str) -> str:
    plan_block   = f"{PLAN_LVR_START}{LVR * n_plan}{PLAN_LVR_END}"
    bbox_block   = f"{BBOX_LVR_START}{LVR * 3}{BBOX_LVR_END}"
    patch_block  = f"{PATCH_LVR_START}{LVR * n_patch}{PATCH_LVR_END}"
    answer_block = f"{FULL_LVR_START}{LVR * n_answer}{FULL_LVR_END}"
    return (
        f"{plan_block}{bbox_block}{patch_block}{answer_block}"
        f"{ANSWER_OPEN} {short_answer.strip()} {ANSWER_CLOSE}"
    )


class GqaCotDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor: Any,
        tokenizer: Any,
        max_seq_length: int = 4096,
        max_plan_tokens: int = 128,
        n_patch_tokens: int = 0,
        max_full_answer_tokens: int = 48,
        patch_fraction: float = 0.5,
        min_patches: int = 5,
    ):
        self.image_folder = image_folder
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.max_plan_tokens = int(max_plan_tokens)
        self.max_patch_tokens = int(n_patch_tokens)
        self.max_full_answer_tokens = int(max_full_answer_tokens)
        self.patch_fraction = float(patch_fraction)
        self.min_patches = int(min_patches)

        self.image_token_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.lvr_id       = int(tokenizer.convert_tokens_to_ids(LVR))

        self.rows = self._load(data_path)

    @staticmethod
    def _load(path: str) -> List[dict]:
        if path.endswith(".jsonl"):
            rows: List[dict] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        question    = row["question"]
        short_ans   = row.get("answer", row.get("short_answer", "")).strip()
        full_ans    = row.get("full_answer", "")
        if not isinstance(full_ans, str) or not full_ans.strip():
            raise ValueError(
                f"Sample {idx} ({row.get('image', '<no image>')}) is missing required "
                "`full_answer`; do not fall back to `thought` for the full-answer block."
            )
        full_ans    = full_ans.strip()
        cot_text    = row.get("thought", "").strip()
        if not cot_text:
            raise ValueError(
                f"Sample {idx} ({row.get('image', '<no image>')}) is missing required "
                "`thought` for the action-plan block."
            )
        image_name  = row["image"]
        bboxs       = row.get("bboxs", [])
        w = int(row.get("width",  1))
        h = int(row.get("height", 1))

        img_path = os.path.join(self.image_folder, image_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))

        # BBox → normalized xyxy → cxcywh
        bbox_raw  = bboxs[0] if bboxs else [0, 0, max(w, 1), max(h, 1)]
        bbox_norm = _normalize_bbox(bbox_raw, w, h)
        cx, cy, bw, bh = _xyxy_to_cxcywh(*bbox_norm)
        bbox_coords = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        # Tokenize CoT (plan) → K1
        pad_id = self.tokenizer.pad_token_id or 0
        cot_enc = self.tokenizer(
            cot_text, add_special_tokens=False, return_tensors="pt",
            truncation=True, max_length=self.max_plan_tokens,
        )
        cot_ids   = cot_enc["input_ids"][0].long()
        n_plan    = max(1, min(int(cot_ids.numel()), self.max_plan_tokens))

        # Tokenize full answer → K4
        ans_enc = self.tokenizer(
            full_ans, add_special_tokens=False, return_tensors="pt",
            truncation=True, max_length=self.max_full_answer_tokens,
        )
        ans_ids   = ans_enc["input_ids"][0].long()
        n_answer  = max(1, min(int(ans_ids.numel()), self.max_full_answer_tokens))

        # Build chat text
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": question},
            ]}
        ]
        text_prefix = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prefix_inputs = self.processor(
            text=[text_prefix], images=[img], return_tensors="pt",
            truncation=False, padding=False,
        )
        prefix_grid = prefix_inputs.get("image_grid_thw")
        n_patch = self.min_patches
        if isinstance(prefix_grid, torch.Tensor) and prefix_grid.numel() >= 3:
            grid_0 = prefix_grid[0] if prefix_grid.dim() == 2 else prefix_grid
            try:
                bbox_idxs = bbox_to_token_idxs_lvr_packed(tuple(bbox_norm), grid_0)
                n_bbox_patch = max(1, len(bbox_idxs))
                n_patch = max(self.min_patches, int(math.ceil(self.patch_fraction * n_bbox_patch)))
                n_patch = min(n_patch, n_bbox_patch)
            except Exception:
                n_patch = self.min_patches
        if self.max_patch_tokens > 0:
            n_patch = min(n_patch, self.max_patch_tokens)
        n_patch = max(1, int(n_patch))

        assistant = _build_assistant_text(n_plan, n_patch, n_answer, short_ans)
        text_full = text_prefix + assistant + (self.tokenizer.eos_token or "")

        inputs = self.processor(
            text=[text_full], images=[img], return_tensors="pt",
            truncation=False, padding=False,
        )
        out: Dict[str, Any] = {
            k: v.squeeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        ids = out["input_ids"]

        # Ensure image_grid_thw is 2D
        if "image_grid_thw" in out and out["image_grid_thw"].dim() == 1:
            out["image_grid_thw"] = out["image_grid_thw"].unsqueeze(0)

        # Labels: CE on everything in assistant turn except <lvr> placeholder tokens
        labels = ids.clone()
        asst_start = prefix_inputs["input_ids"].shape[1]
        if asst_start > 0:
            labels[:asst_start] = IGNORE_INDEX
        if "attention_mask" in out:
            labels = labels.masked_fill(out["attention_mask"] == 0, IGNORE_INDEX)
        # Mask <lvr> placeholder tokens from CE — they're MSE-supervised in forward
        labels = labels.masked_fill(ids == self.lvr_id, IGNORE_INDEX)

        # Pad plan token ids to max_plan_tokens
        cot_padded = torch.full((self.max_plan_tokens,), pad_id, dtype=torch.long)
        store_plan = min(n_plan, int(cot_ids.numel()))
        if store_plan > 0:
            cot_padded[:store_plan] = cot_ids[:store_plan]

        # Pad full-answer token ids to max_full_answer_tokens
        ans_padded = torch.full((self.max_full_answer_tokens,), pad_id, dtype=torch.long)
        store_ans = min(n_answer, int(ans_ids.numel()))
        if store_ans > 0:
            ans_padded[:store_ans] = ans_ids[:store_ans]

        out["labels"]               = labels
        out["cot_token_ids"]        = cot_padded
        out["cot_token_num"]        = torch.tensor(store_plan,  dtype=torch.long)
        out["answer_token_ids"]     = ans_padded
        out["answer_token_num"]     = torch.tensor(store_ans,   dtype=torch.long)
        out["bbox_coords"]          = bbox_coords          # [4]: cx, cy, w, h
        out["bbox_norm"]            = torch.tensor(list(bbox_norm), dtype=torch.float32)  # [4]: x0,y0,x1,y1
        return out


@dataclass
class GqaCotCollator:
    """Pads variable-length tensor fields for batching."""
    pad_token_id: int

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = list(examples[0].keys())
        sample_lens = [int(ex["input_ids"].numel()) for ex in examples]
        max_len = max(sample_lens) if sample_lens else 0

        pad_overrides = {
            "labels": IGNORE_INDEX,
            "attention_mask": 0,
            "token_type_ids": 0,
        }
        non_aligned = {
            "cot_token_ids", "answer_token_ids", "image_grid_thw", "pixel_values",
            "bbox_coords", "bbox_norm",
        }

        batch: Dict[str, Any] = {}
        for k in keys:
            v0 = examples[0][k]
            if not isinstance(v0, torch.Tensor):
                batch[k] = [ex[k] for ex in examples]
                continue

            aligned = (
                v0.ndim == 1
                and k not in non_aligned
                and all(int(ex[k].numel()) == sample_lens[i] for i, ex in enumerate(examples))
            )
            if aligned:
                pv = pad_overrides.get(k, self.pad_token_id)
                rows = []
                for ex in examples:
                    t = ex[k]
                    p = max_len - int(t.numel())
                    if p > 0:
                        pad = t.new_full((p,), bool(pv) if t.dtype == torch.bool else pv)
                        t = torch.cat([t, pad])
                    rows.append(t)
                batch[k] = torch.stack(rows)
                continue
            if k == "image_grid_thw":
                batch[k] = torch.cat([ex[k] for ex in examples], dim=0)
                continue
            if k == "pixel_values":
                batch[k] = torch.cat([ex[k] for ex in examples], dim=0)
                continue
            try:
                batch[k] = torch.stack([ex[k] for ex in examples])
            except RuntimeError:
                batch[k] = [ex[k] for ex in examples]

        if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
            batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()
        return batch
