"""MCQA Dataset for visual grounding pretraining on Qwen2.5-VL.

Each sample: image + masked/re-written caption + options → answer letter.
Loss is computed only on the answer token.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100

SYSTEM_MESSAGE_MCQA = (
    "Look at the image carefully and choose the most accurate description "
    "from the options below. Answer with a single letter."
)


class MCQADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor: Any,
        tokenizer: Any,
        max_seq_length: int = 2048,
    ):
        self.image_folder = image_folder
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_name = row["image"]
        img_path = os.path.join(self.image_folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))

        context = row.get("rewritten_caption", "")
        answer_letter = row.get("answer", "A")
        options = []
        for letter in ("A", "B", "C", "D"):
            value = row.get(f"option_{letter}")
            if value:
                options.append((letter, value))
        if not row.get("option_D"):
            options.append(("D", "I don't know"))
        option_text = "\n".join(f"{letter}. {value}" for letter, value in options)

        user_text = (
            f"{context}\n\n"
            f"Which of the following best describes what you see in the image?\n"
            f"{option_text}"
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE_MCQA}]},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": user_text},
            ]},
        ]
        text_prefix = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text_full = text_prefix + answer_letter + (self.tokenizer.eos_token or "")

        inputs = self.processor(
            text=[text_full], images=[img], return_tensors="pt",
            truncation=True, max_length=self.max_seq_length, padding=False,
        )
        out: Dict[str, Any] = {
            k: v.squeeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        ids = out["input_ids"]

        prefix_inputs = self.processor(
            text=[text_prefix], images=[img], return_tensors="pt",
            truncation=True, max_length=self.max_seq_length, padding=False,
        )
        asst_start = prefix_inputs["input_ids"].shape[1]

        labels = ids.clone()
        labels[:asst_start] = IGNORE_INDEX
        if "attention_mask" in out:
            labels = labels.masked_fill(out["attention_mask"] == 0, IGNORE_INDEX)

        if "image_grid_thw" in out and out["image_grid_thw"].dim() == 1:
            out["image_grid_thw"] = out["image_grid_thw"].unsqueeze(0)

        out["labels"] = labels
        return out


@dataclass
class MCQACollator:
    pad_token_id: int

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = list(examples[0].keys())
        sample_lens = [int(ex["input_ids"].numel()) for ex in examples]
        max_len = max(sample_lens) if sample_lens else 0

        pad_overrides = {"labels": IGNORE_INDEX, "attention_mask": 0}
        non_aligned = {"image_grid_thw", "pixel_values"}

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
