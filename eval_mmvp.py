#!/usr/bin/env python3
"""MMVP + V*Bench eval for PinReason v2 on Qwen2.5-VL.

Supports:
  --mode pinreason   two-stage LVR decode
  --mode baseline    standard Qwen2.5-VL generation
  --bench mmvp / vstar / all
  --ablation         sweep over lvr1/lvr2 step counts
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import *

_DEFAULT_MMVP_CSV = Path("/path/to/MMVP/Questions.csv")
_DEFAULT_MMVP_IMAGES = Path("/path/to/MMVP/MMVP Images")
_DEFAULT_VSTAR_JSONL = Path("/path/to/vstar_bench/test_questions.jsonl")
_DEFAULT_VSTAR_ROOT = Path("/path/to/vstar_bench")


# ===================================================================
#  Data loaders
# ===================================================================

def _parse_options(opts):
    m = re.match(r"\(a\)\s*(.+?)\s*\(b\)\s*(.+)", opts, re.IGNORECASE)
    return (m.group(1).strip().lower(), m.group(2).strip().lower()) if m else ("", "")


def extract_letter_ab(pred, opt_a="", opt_b=""):
    if not pred:
        return None
    s = pred.strip().lower()
    if s and s[0] in ("a", "b"):
        return s[0]
    m = re.search(r"\(([ab])\)", s)
    if m:
        return m.group(1)
    if opt_a and opt_b:
        if opt_a in s and opt_b not in s:
            return "a"
        if opt_b in s and opt_a not in s:
            return "b"
    m = re.search(r"\b([ab])\b", s)
    return m.group(1) if m else None


def extract_letter_abcd(pred):
    if not pred:
        return None
    s = pred.strip().upper()
    if s and s[0] in "ABCD":
        return s[0]
    m = re.search(r"\(([ABCD])\)", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", s)
    return m.group(1) if m else None


def load_mmvp(csv_path, img_dir):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx = r["Index"].strip()
            gt = "a" if "(a)" in r["Correct Answer"].strip().lower() else "b"
            opts = r["Options"].strip()
            oa, ob = _parse_options(opts)
            img = img_dir / f"{idx}.jpg"
            if not img.is_file():
                img = img_dir / f"{idx}.png"
            rows.append({"bench": "mmvp", "id": idx,
                         "question": f"{r['Question'].strip()}\n{opts}\nReply with only (a) or (b).",
                         "image_path": str(img), "answer": gt, "opt_a": oa, "opt_b": ob})
    return rows


def load_vstar(jsonl_path, root):
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            img = root / item["image"]
            if not img.is_file():
                continue
            rows.append({"bench": "vstar", "id": item.get("question_id", ""),
                         "category": item.get("category", ""),
                         "question": item["text"].strip() + "\nAnswer with only one letter (A/B/C/D).",
                         "image_path": str(img), "answer": item["label"].strip().upper(),
                         "opt_a": "", "opt_b": ""})
    print(f"[V*Bench] {len(rows)} samples")
    return rows


# ===================================================================
#  PinReason v2 decode
# ===================================================================

@torch.inference_mode()
def decode_pinreason(model, processor, tokenizer, image_path, question,
                     device, dtype, *,
                     n_plan=64, n_patch=16, n_answer=64, answer_tokens=32,
                     choice_options=None, choice_scoring=False, use_stage_text=True):
    """4-step latent reasoning inference.

    Each <lvr> block uses hidden-state feedback (like LVR):
        step 0  : input = hidden of last text token before block
        step t>0: input = hidden of previous latent step
    Typed stage-start/end tokens are fed explicitly.
    """
    pil = Image.open(image_path).convert("RGB")
    system_text = "Answer the question with exactly one option letter. Do not explain."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": system_text + "\n" + question},
    ]}]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text_prompt], images=[pil], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    if inputs.get("pixel_values") is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    _base   = model.module if hasattr(model, "module") else model
    core    = _base.model
    lm_head = _base.lm_head
    _dtype  = next(core.language_model.parameters()).dtype
    eos_id  = int(tokenizer.eos_token_id)
    answer_close_ids = set(tokenizer(ANSWER_CLOSE, add_special_tokens=False)["input_ids"])

    # ── Prefill ──────────────────────────────────────────────────────────────
    out     = model(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    h       = out.hidden_states[-1][:, -1, :].detach()   # [1, H]
    cur_len = inputs["input_ids"].shape[1]

    def _make_pos(n_tok):
        cache_pos   = torch.arange(cur_len, cur_len + n_tok, device=device, dtype=torch.long)
        rope_deltas = getattr(core, "rope_deltas", None)
        pos = torch.arange(n_tok, device=device).view(1, 1, -1).expand(3, 1, -1).contiguous()
        offset = (cur_len + rope_deltas.to(device).view(1, 1, 1)
                  if rope_deltas is not None else cur_len)
        return cache_pos, (pos + offset).long()

    def _step(*, input_ids=None, inputs_embeds=None, need_hidden=False, n_tok=1):
        nonlocal past_kv, cur_len
        cache_pos, position_ids = _make_pos(n_tok)
        t_out = core.language_model(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            past_key_values=past_kv, use_cache=True,
            output_hidden_states=need_hidden,
            position_ids=position_ids,
            cache_position=cache_pos,
        )
        past_kv  = t_out.past_key_values
        cur_len += n_tok
        logits   = lm_head(t_out.last_hidden_state[:, -1:, :])
        hs       = t_out.hidden_states[-1][:, -1, :].detach() if need_hidden else None
        return logits, hs

    def _feed_text(text: str):
        """Feed a fixed text string token-by-token, return last hidden state."""
        nonlocal h
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = ids.to(device).squeeze(0)
        for tid in ids:
            _, h = _step(input_ids=tid.view(1, 1), need_hidden=True)
        return h

    def _latent_block(K: int):
        """K steps of hidden-state feedback. Returns list of hidden states."""
        nonlocal h
        all_h = []
        for _ in range(K):
            emb = h.to(_dtype).unsqueeze(0)   # [1, 1, H]
            _, h = _step(inputs_embeds=emb, need_hidden=True)
            all_h.append(h.squeeze(0))
        return all_h

    # ── Block 1: action plan ─────────────────────────────────────────────────
    _feed_text(PLAN_LVR_START)
    _latent_block(n_plan)
    _feed_text(PLAN_LVR_END)

    # ── Block 2: bounding box ────────────────────────────────────────────────
    _feed_text(BBOX_LVR_START)
    _latent_block(3)
    _feed_text(BBOX_LVR_END)

    # ── Block 3: embedding token ─────────────────────────────────────────────
    _feed_text(PATCH_LVR_START)
    _latent_block(n_patch)
    _feed_text(PATCH_LVR_END)

    # ── Block 4: full answer ─────────────────────────────────────────────────
    _feed_text(FULL_LVR_START)
    _latent_block(n_answer)
    _feed_text(FULL_LVR_END)

    # ── AR decode: <answer> short_answer </answer> ───────────────────────────
    answer_prefix_ids = tokenizer(f"{ANSWER_OPEN} ", add_special_tokens=False)["input_ids"]
    logits, _ = _step(
        input_ids=torch.tensor(answer_prefix_ids, device=device).view(1, -1),
        n_tok=len(answer_prefix_ids),
    )

    if choice_scoring and choice_options:
        scores = {}
        saved_past, saved_h, saved_len = past_kv, h, cur_len
        for label, text in choice_options.items():
            past_kv, h, cur_len = saved_past, saved_h, saved_len
            cand_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
            cand_ids = cand_ids.to(device).squeeze(0)
            if cand_ids.numel() == 0:
                scores[label] = float("-inf")
                continue
            score = logits[:, -1, :].float().log_softmax(-1)[0, cand_ids[0]].item()
            for i, tid in enumerate(cand_ids[:-1]):
                next_logits, _ = _step(input_ids=tid.view(1, 1), need_hidden=False)
                score += next_logits[:, -1, :].float().log_softmax(-1)[0, cand_ids[i + 1]].item()
            scores[label] = score / max(int(cand_ids.numel()), 1)
        past_kv, h, cur_len = saved_past, saved_h, saved_len
        best = max(scores, key=scores.get)
        return {
            "answer_text": choice_options[best],
            "pred_letter": best,
            "choice_scores": scores,
            "n_plan": n_plan, "n_patch": n_patch, "n_answer": n_answer,
        }

    nid = int(logits[:, -1, :].argmax(-1).item())
    gen = [nid]
    for _ in range(answer_tokens - 1):
        if nid == eos_id or nid in answer_close_ids:
            break
        logits, _ = _step(input_ids=torch.tensor([[nid]], device=device))
        nid = int(logits[:, -1, :].argmax(-1).item())
        gen.append(nid)

    return {
        "answer_text": tokenizer.decode(gen, skip_special_tokens=True).strip(),
        "answer_text_raw": tokenizer.decode(gen, skip_special_tokens=False).strip(),
        "answer_token_ids": gen,
        "n_plan": n_plan, "n_patch": n_patch, "n_answer": n_answer,
    }


@torch.inference_mode()
def decode_baseline(model, processor, tokenizer, image_path, question,
                    device, dtype, *, max_new_tokens=64):
    pil = Image.open(image_path).convert("RGB")
    system_text = "Answer the question with exactly one option letter. Do not explain."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": system_text + "\n" + question},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[pil], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    if inputs.get("pixel_values") is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return {"answer_text": tokenizer.decode(gen_ids, skip_special_tokens=True).strip()}


# ===================================================================
#  Scoring
# ===================================================================

def score_mmvp(results):
    n_ok = sum(1 for r in results if r.get("correct"))
    n = len(results)
    pair_ok = sum(1 for i in range(0, n - 1, 2)
                  if results[i].get("correct") and results[i + 1].get("correct"))
    return {"n": n, "correct": n_ok, "individual_acc": n_ok / max(n, 1),
            "pairs": n // 2, "pair_correct": pair_ok, "pair_acc": pair_ok / max(n // 2, 1)}


def score_vstar(results):
    n = len(results)
    n_ok = sum(1 for r in results if r.get("correct"))
    cats: dict = {}
    for r in results:
        c = r.get("category", "all")
        cats.setdefault(c, {"n": 0, "ok": 0})
        cats[c]["n"] += 1
        if r.get("correct"):
            cats[c]["ok"] += 1
    s = {"n": n, "correct": n_ok, "accuracy": n_ok / max(n, 1)}
    for c, st in sorted(cats.items()):
        s[f"cat_{c}_acc"] = st["ok"] / max(st["n"], 1)
        s[f"cat_{c}_n"] = st["n"]
    return s


# ===================================================================
#  Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--mode", choices=("pinreason", "baseline"), default="pinreason")
    ap.add_argument("--bench", choices=("mmvp", "vstar", "all"), default="all")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_plan",    type=int, default=64,  help="plan latent steps")
    ap.add_argument("--n_patch",   type=int, default=16,  help="patch latent steps")
    ap.add_argument("--n_answer",  type=int, default=64,  help="answer latent steps")
    ap.add_argument("--answer_tokens", type=int, default=32)
    ap.add_argument("--mmvp_csv", type=Path, default=_DEFAULT_MMVP_CSV)
    ap.add_argument("--mmvp_images", type=Path, default=_DEFAULT_MMVP_IMAGES)
    ap.add_argument("--vstar_jsonl", type=Path, default=_DEFAULT_VSTAR_JSONL)
    ap.add_argument("--vstar_root", type=Path, default=_DEFAULT_VSTAR_ROOT)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(ROOT / "results" / f"{Path(args.model_path).name}_{args.mode}")
    out_dir = Path(args.out_dir)

    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    is_pr = args.mode == "pinreason"

    print(f"Loading {args.model_path} ...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if is_pr:
        for t in SPECIAL_TOKENS:
            tokenizer.add_tokens(t, special_tokens=True)
        from src.forward import install_pinreason_v2_patches
        install_pinreason_v2_patches(tokenizer=tokenizer)

    device = torch.device("cuda")
    dt = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=dt,
        attn_implementation="eager", trust_remote_code=True,
    ).to(device).eval()
    if is_pr:
        model.resize_token_embeddings(len(tokenizer))
    print(f"Loaded on {device} | mode={args.mode}")

    bench_data = {}
    if args.bench in ("mmvp", "all"):
        bench_data["mmvp"] = load_mmvp(args.mmvp_csv, args.mmvp_images)
    if args.bench in ("vstar", "all"):
        bench_data["vstar"] = load_vstar(args.vstar_jsonl, args.vstar_root)

    tag = f"plan={args.n_plan}_patch={args.n_patch}_ans={args.n_answer}" if is_pr else "baseline"

    all_summaries = []
    for bench, samples in bench_data.items():
        results = []
        pred_path = out_dir / f"{bench}_{tag}_predictions.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as fout:
            for s in tqdm(samples, desc=f"{bench} [{tag}]"):
                try:
                    r = (decode_pinreason(model, processor, tokenizer,
                                          s["image_path"], s["question"], device, dt,
                                          n_plan=args.n_plan, n_patch=args.n_patch,
                                          n_answer=args.n_answer,
                                          answer_tokens=args.answer_tokens)
                         if is_pr else
                         decode_baseline(model, processor, tokenizer,
                                         s["image_path"], s["question"], device, dt,
                                         max_new_tokens=args.answer_tokens))
                    if bench == "mmvp":
                        r["pred_letter"] = extract_letter_ab(r["answer_text"], s.get("opt_a"), s.get("opt_b"))
                        r["correct"] = r["pred_letter"] == s["answer"]
                    else:
                        r["pred_letter"] = extract_letter_abcd(r["answer_text"])
                        r["correct"] = r["pred_letter"] == s["answer"]
                    rec = {**s, **r}
                except Exception as e:
                    rec = {**s, "answer_text": None, "correct": False,
                           "error": str(e), "tb": traceback.format_exc()}
                results.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        scores = score_mmvp(results) if bench == "mmvp" else score_vstar(results)
        scores.update(bench=bench, mode=args.mode, config=tag)
        with open(out_dir / f"{bench}_{tag}_summary.json", "w") as fp:
            json.dump(scores, fp, indent=2)
        all_summaries.append(scores)

        print(f"\n{'='*60}\n{bench.upper()} | {tag}")
        if bench == "mmvp":
            print(f"  Acc: {scores['individual_acc']:.4f}  Pair: {scores['pair_acc']:.4f}")
        else:
            print(f"  Acc: {scores['accuracy']:.4f}")
        print(f"{'='*60}")

    with open(out_dir / "combined_summary.json", "w") as fp:
        json.dump(all_summaries, fp, indent=2)
    print(f"\nAll results → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
