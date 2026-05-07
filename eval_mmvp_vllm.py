#!/usr/bin/env python3
"""MMVP + V*Bench with vLLM (multi-GPU tensor parallel) — avoids single-GPU OOM on large VL images.

Does NOT use Hugging Face model.generate(); uses Qwen-VL multimodal prompts like generate_mcqa.py.

**Env pitfall**: vLLM workers import PyTorch; if conda has ``torch...+cuXXX`` newer than your
``nvidia-smi`` driver's supported CUDA runtime, you get ``torch._cuda_init`` / driver-too-old inside
workers. Fix by installing a cu12 wheel that matches the driver + a matching vLLM, or upgrading the
GPU driver.


Usage:
  PYTHONPATH=. python eval_mmvp_vllm.py \\
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \\
    --tensor_parallel 4 \\
    --bench vstar
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from eval_mmvp import (
    _DEFAULT_MMVP_CSV,
    _DEFAULT_MMVP_IMAGES,
    _DEFAULT_VSTAR_JSONL,
    _DEFAULT_VSTAR_ROOT,
    extract_letter_ab,
    extract_letter_abcd,
    load_mmvp,
    load_vstar,
    score_mmvp,
    score_vstar,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=Path)
    ap.add_argument("--bench", choices=("mmvp", "vstar", "all"), default="all")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="Must cover text + vision tokens; V* at default max_pixels can exceed 8k.",
    )
    ap.add_argument("--max_tokens", type=int, default=64, help="Max new tokens to generate.")
    ap.add_argument("--max_pixels", type=int, default=401408)
    ap.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mmvp_csv", type=Path, default=_DEFAULT_MMVP_CSV)
    ap.add_argument("--mmvp_images", type=Path, default=_DEFAULT_MMVP_IMAGES)
    ap.add_argument("--vstar_jsonl", type=Path, default=_DEFAULT_VSTAR_JSONL)
    ap.add_argument("--vstar_root", type=Path, default=_DEFAULT_VSTAR_ROOT)
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ROOT / "results" / f"{args.model_path.name}_vllm_{args.bench}"

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    print(f"vLLM loading {args.model_path} | tp={args.tensor_parallel} ...")
    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        max_pixels=args.max_pixels,
    )

    bench_data: dict = {}
    if args.bench in ("mmvp", "all"):
        bench_data["mmvp"] = load_mmvp(args.mmvp_csv, args.mmvp_images)
    if args.bench in ("vstar", "all"):
        bench_data["vstar"] = load_vstar(args.vstar_jsonl, args.vstar_root)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "baseline"
    all_summaries = []

    for bench, samples in bench_data.items():
        results = []
        pred_path = out_dir / f"{bench}_{tag}_predictions.jsonl"
        prompts = []

        # Build multimodal payloads (same idea as Gemma mcqa generator).
        meta = []
        for s in samples:
            img_path = s["image_path"]
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": s["question"]},
                ]}
            ]
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            # vLLM parses `multi_modal_data["image"]` as a sequence of items; a bare
            # filesystem path string is iterated character-by-character (first "image" → "/").
            prompts.append({"prompt": text_prompt, "multi_modal_data": {"image": [img_path]}})
            meta.append(s)

        print(f"Inference [{bench}] n={len(prompts)} batches (vLLM internal batching) ...")

        outputs = llm.generate(prompts, sampling_params=sampling)
        assert len(outputs) == len(meta)

        for s, vo in zip(meta, tqdm(outputs, total=len(outputs), desc=f"{bench} write")):
            raw = vo.outputs[0].text
            try:
                if bench == "mmvp":
                    pred_letter = extract_letter_ab(raw, s.get("opt_a"), s.get("opt_b"))
                    correct = pred_letter == s["answer"]
                else:
                    pred_letter = extract_letter_abcd(raw)
                    correct = pred_letter == s["answer"]
                rec = {**s, "answer_text": raw.strip(), "pred_letter": pred_letter, "correct": correct}
            except Exception as e:
                rec = {
                    **s,
                    "answer_text": raw,
                    "correct": False,
                    "pred_letter": None,
                    "error": str(e),
                    "tb": traceback.format_exc(),
                }
            results.append(rec)

        with open(pred_path, "w", encoding="utf-8") as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        scores = score_mmvp(results) if bench == "mmvp" else score_vstar(results)
        scores.update(bench=bench, mode="baseline_vllm", config=f"tp={args.tensor_parallel}")
        out_json = out_dir / f"{bench}_{tag}_summary.json"
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(scores, fp, indent=2)
        print(f"[{bench}] " + json.dumps(scores, ensure_ascii=False)[:800])
        all_summaries.append(scores)

    with open(out_dir / "combined_summary.json", "w", encoding="utf-8") as fp:
        json.dump(all_summaries, fp, indent=2)
    print(f"Done → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
