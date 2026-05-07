#!/usr/bin/env python3
"""Download CoVT-Dataset part6 and export images + metadata to disk."""

import os
import json
from datasets import load_dataset
from PIL import Image

SAVE_DIR = "/path/to/covt_part6"
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
META_PATH = os.path.join(SAVE_DIR, "metadata.jsonl")

os.makedirs(IMAGE_DIR, exist_ok=True)

print("Downloading Wakals/CoVT-Dataset part6 ...")
ds = load_dataset("Wakals/CoVT-Dataset", "part6", split="train")
print(f"Loaded {len(ds)} rows.")

with open(META_PATH, "w", encoding="utf-8") as f_out:
    for i, row in enumerate(ds):
        img_name = f"{row.get('id', i)}.png"
        img_path = os.path.join(IMAGE_DIR, img_name)

        img = row.get("image")
        if img is not None and not os.path.exists(img_path):
            if isinstance(img, Image.Image):
                img.save(img_path)
            else:
                Image.open(img).save(img_path)

        meta = {
            "id": row.get("id", str(i)),
            "image": img_name,
            "conversations": row.get("conversations", []),
        }
        f_out.write(json.dumps(meta, ensure_ascii=False) + "\n")

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(ds)}] saved")

print(f"Done. Images → {IMAGE_DIR}, Metadata → {META_PATH}")
