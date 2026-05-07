#!/usr/bin/env python3
"""Generic evaluation entrypoint for PinReason v2 and Qwen2.5-VL baselines."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_mmvp import (
    decode_baseline,
    decode_pinreason,
    extract_letter_ab,
    extract_letter_abcd,
    score_mmvp,
    score_vstar,
)
from src.constants import SPECIAL_TOKENS

DEFAULT_MMVP_CSV = Path("/path/to/MMVP/Questions.csv")
DEFAULT_MMVP_IMAGES = Path("/path/to/MMVP/MMVP Images")
DEFAULT_VSTAR_JSONL = Path("/path/to/vstar_bench/test_questions.jsonl")
DEFAULT_VSTAR_ROOT = Path("/path/to/vstar_bench")


def _parse_options(opts: str) -> tuple[str, str]:
    m = re.match(r"\(a\)\s*(.+?)\s*\(b\)\s*(.+)", opts, re.IGNORECASE)
    return (m.group(1).strip().lower(), m.group(2).strip().lower()) if m else ("", "")


def load_dataset_rows(dataset: str, data_path: Path, image_root: Path) -> list[dict]:
    if dataset == "mmvp":
        rows = []
        with open(data_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                idx = r["Index"].strip()
                gt = "a" if "(a)" in r["Correct Answer"].strip().lower() else "b"
                opts = r["Options"].strip()
                oa, ob = _parse_options(opts)
                img = image_root / f"{idx}.jpg"
                if not img.is_file():
                    img = image_root / f"{idx}.png"
                rows.append({
                    "dataset": "mmvp",
                    "id": idx,
                    "question": (
                        f"{r['Question'].strip()}\n{opts}\n"
                        "Answer with exactly one option letter: A or B."
                    ),
                    "image_path": str(img),
                    "answer": gt,
                    "opt_a": oa,
                    "opt_b": ob,
                })
        return rows

    if dataset == "vstar":
        rows = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                img = image_root / item["image"]
                if not img.is_file():
                    continue
                rows.append({
                    "dataset": "vstar",
                    "id": item.get("question_id", ""),
                    "category": item.get("category", ""),
                    "question": item["text"].strip() + "\nAnswer with only one letter (A/B/C/D).",
                    "image_path": str(img),
                    "answer": item["label"].strip().upper(),
                    "opt_a": "",
                    "opt_b": "",
                })
        return rows

    raise ValueError(f"unsupported dataset: {dataset}")


def score_dataset(dataset: str, results: list[dict]) -> dict:
    if dataset == "mmvp":
        return score_mmvp(results)
    if dataset == "vstar":
        return score_vstar(results)
    raise ValueError(f"unsupported dataset: {dataset}")


def add_prediction(dataset: str, record: dict, decoded: dict) -> dict:
    if dataset == "mmvp":
        decoded["pred_letter"] = extract_letter_ab(
            decoded["answer_text"], record.get("opt_a"), record.get("opt_b")
        )
        decoded["correct"] = decoded["pred_letter"] == record["answer"]
    elif dataset == "vstar":
        decoded["pred_letter"] = extract_letter_abcd(decoded["answer_text"])
        decoded["correct"] = decoded["pred_letter"] == record["answer"]
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return {**record, **decoded}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--mode", choices=("pinreason", "baseline"), default="pinreason")
    ap.add_argument("--dataset", choices=("mmvp", "vstar", "all"), required=True)
    ap.add_argument("--data_path", type=Path, default=None)
    ap.add_argument("--image_root", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--n_plan", type=int, default=64)
    ap.add_argument("--n_patch", type=int, default=16)
    ap.add_argument("--n_full_answer", type=int, default=8)
    ap.add_argument("--answer_tokens", type=int, default=32)
    ap.add_argument("--choice_scoring", action="store_true")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.dataset != "all":
        if args.data_path is None:
            args.data_path = DEFAULT_MMVP_CSV if args.dataset == "mmvp" else DEFAULT_VSTAR_JSONL
        if args.image_root is None:
            args.image_root = DEFAULT_MMVP_IMAGES if args.dataset == "mmvp" else DEFAULT_VSTAR_ROOT
    elif args.data_path is not None or args.image_root is not None:
        raise ValueError("--data_path/--image_root are only supported when --dataset is mmvp or vstar")
    if args.out_dir is None:
        args.out_dir = ROOT / "results" / f"{Path(args.model_path).name}_{args.mode}_{args.dataset}"

    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    is_pinreason = args.mode == "pinreason"
    print(f"Loading model: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if is_pinreason:
        for token in SPECIAL_TOKENS:
            tokenizer.add_tokens(token, special_tokens=True)
        from src.forward import install_pinreason_v2_patches
        install_pinreason_v2_patches(tokenizer=tokenizer)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()
    if is_pinreason:
        model.resize_token_embeddings(len(tokenizer))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"plan={args.n_plan}_patch={args.n_patch}_full={args.n_full_answer}"
        if is_pinreason else "baseline"
    )
    datasets = ["mmvp", "vstar"] if args.dataset == "all" else [args.dataset]
    combined = []

    for dataset in datasets:
        data_path = args.data_path
        image_root = args.image_root
        if dataset == "mmvp":
            data_path = data_path or DEFAULT_MMVP_CSV
            image_root = image_root or DEFAULT_MMVP_IMAGES
        elif dataset == "vstar":
            data_path = data_path or DEFAULT_VSTAR_JSONL
            image_root = image_root or DEFAULT_VSTAR_ROOT

        rows = load_dataset_rows(dataset, data_path, image_root)
        if args.limit > 0:
            rows = rows[:args.limit]
        pred_ext = "json" if args.pretty else "jsonl"
        pred_path = args.out_dir / f"{dataset}_{tag}_predictions.{pred_ext}"
        summary_path = args.out_dir / f"{dataset}_{tag}_summary.json"

        results = []
        fout = open(pred_path, "w", encoding="utf-8") if not args.pretty else None
        try:
            for row in tqdm(rows, desc=f"{dataset} [{tag}]"):
                try:
                    if is_pinreason:
                        decoded = decode_pinreason(
                            model, processor, tokenizer,
                            row["image_path"], row["question"], device, dtype,
                            n_plan=args.n_plan,
                            n_patch=args.n_patch,
                            n_answer=args.n_full_answer,
                            answer_tokens=args.answer_tokens,
                            choice_options=(
                                {"a": row["opt_a"], "b": row["opt_b"]}
                                if dataset == "mmvp" and row.get("opt_a") and row.get("opt_b")
                                else None
                            ),
                            choice_scoring=args.choice_scoring,
                        )
                    else:
                        decoded = decode_baseline(
                            model, processor, tokenizer,
                            row["image_path"], row["question"], device, dtype,
                            max_new_tokens=args.answer_tokens,
                        )
                    rec = add_prediction(dataset, row, decoded)
                except Exception as exc:
                    rec = {
                        **row,
                        "answer_text": None,
                        "correct": False,
                        "error": str(exc),
                        "tb": traceback.format_exc(),
                    }
                results.append(rec)
                if fout is not None:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            if fout is not None:
                fout.close()

        if args.pretty:
            with open(pred_path, "w", encoding="utf-8") as fp:
                json.dump(results, fp, ensure_ascii=False, indent=2)

        scores = score_dataset(dataset, results)
        scores.update(dataset=dataset, mode=args.mode, config=tag)
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(scores, fp, indent=2)
        combined.append(scores)

        print(f"Results: {pred_path}")
        print(f"Summary: {summary_path}")
        metric = scores.get("individual_acc", scores.get("accuracy", 0.0))
        print(f"{dataset} accuracy: {metric:.4f}")

    if len(combined) > 1:
        combined_path = args.out_dir / f"combined_{tag}_summary.json"
        with open(combined_path, "w", encoding="utf-8") as fp:
            json.dump(combined, fp, indent=2)
        print(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
