#!/usr/bin/env python3
"""Generate MCQA data from CoVT using Gemma-4-31B-it via vLLM (multimodal).

Usage:
  python scripts/generate_mcqa.py \
    --model_path google/gemma-4-31B-it \
    --metadata /path/to/covt_part6/metadata.jsonl \
    --image_dir /path/to/covt_part6/images \
    --output /path/to/covt_part6/mcqa.jsonl \
    --tensor_parallel 8

The script sends each (image, question, answer) triple to the model and asks
it to produce: (1) a vague description, (2) two hard negatives, all in JSON.
It then assembles a 4-choice QA with a shuffled option order.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

PROMPT_TEMPLATE = """\
You are a data-processing assistant. You will be shown an IMAGE and given the original QUESTION and ANSWER about that image.

Your job:
1. VAGUE: Rewrite the answer, removing ALL visual details (colors, counts, specific object names, sizes, positions, materials, textures). Replace them with generic terms. The vague version should be answerable WITHOUT looking at the image.
2. HARD_NEG_1: Create a plausible but WRONG answer by changing 1-2 visual details.
3. HARD_NEG_2: Create another plausible but WRONG answer by changing DIFFERENT visual details from HARD_NEG_1.

CRITICAL RULES for hard negatives:
- Changed details must be CLEARLY DIFFERENT, NOT synonyms or near-synonyms.
  BAD:  "red" → "crimson" (synonym)
  GOOD: "red" → "blue" (clearly different, distinguishable by looking at the image)
- Counts: change to a clearly different number (3→5, NOT 3→several)
- Objects: change to a clearly different object (cat→dog, NOT cat→kitten)
- Positions: change to opposite (left→right, NOT left→leftward)
- The goal: someone who LOOKS at the image can easily rule out wrong options, but someone who ONLY reads text cannot.

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"vague": "...", "hard_neg_1": "...", "hard_neg_2": "..."}}"""


def parse_json_response(text: str) -> Optional[Dict[str, str]]:
    """Extract JSON from model response, tolerating markdown fences."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        if all(k in obj for k in ("vague", "hard_neg_1", "hard_neg_2")):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def build_mcqa_row(
    sample_id: str,
    image_path: str,
    question: str,
    original_answer: str,
    parsed: Dict[str, str],
) -> Dict[str, Any]:
    """Assemble a 4-choice QA row with shuffled options."""
    options = [
        ("correct", original_answer),
        ("hard_neg_1", parsed["hard_neg_1"]),
        ("hard_neg_2", parsed["hard_neg_2"]),
        ("idk", "I don't know."),
    ]
    random.shuffle(options)
    letters = ["A", "B", "C", "D"]
    answer_letter = None
    option_dict = {}
    for i, (tag, text) in enumerate(options):
        option_dict[f"option_{letters[i]}"] = text
        if tag == "correct":
            answer_letter = letters[i]

    return {
        "id": sample_id,
        "image": image_path,
        "question": question,
        "vague": parsed["vague"],
        **option_dict,
        "answer": answer_letter,
        "original_answer": original_answer,
    }


def extract_qa_from_conversations(convs: List[Dict]) -> List[Dict[str, str]]:
    """Extract (question, answer) pairs from LLaVA-style conversations."""
    pairs = []
    for i in range(0, len(convs) - 1, 2):
        if convs[i].get("from") == "human" and convs[i + 1].get("from") == "gpt":
            q = convs[i]["value"].replace("<image>\n", "").replace("<image>", "").strip()
            a = convs[i + 1]["value"].strip()
            if q and a and len(a) > 5:
                pairs.append({"question": q, "answer": a})
    return pairs


def run_vllm(args):
    from vllm import LLM, SamplingParams

    print(f"Loading model {args.model_path} with tp={args.tensor_parallel} ...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    print(f"Loading metadata from {args.metadata} ...")
    rows = []
    with open(args.metadata, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  {len(rows)} rows loaded.")

    tasks = []
    for row in rows:
        convs = row.get("conversations", [])
        pairs = extract_qa_from_conversations(convs)
        if not pairs:
            continue
        qa = pairs[0]
        img_name = row.get("image", row.get("image_relpath", ""))
        img_path = os.path.join(args.image_dir, img_name) if args.image_dir else img_name
        if not os.path.isfile(img_path):
            continue
        tasks.append({
            "id": row.get("id", img_name),
            "image_path": img_path,
            "image_name": img_name,
            "question": qa["question"],
            "answer": qa["answer"],
        })

    print(f"  {len(tasks)} valid tasks with images.")

    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
        print(f"  Capped to {len(tasks)} samples.")

    prompts = []
    for t in tasks:
        text = PROMPT_TEMPLATE.format(question=t["question"], answer=t["answer"])
        prompts.append({
            "prompt": text,
            "multi_modal_data": {"image": t["image_path"]},
        })

    print(f"Running inference on {len(prompts)} prompts ...")
    outputs = llm.generate(prompts, sampling_params=sampling)

    n_ok = 0
    n_fail = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for task, out in zip(tasks, outputs):
            text = out.outputs[0].text
            parsed = parse_json_response(text)
            if parsed is None:
                n_fail += 1
                continue
            mcqa = build_mcqa_row(
                sample_id=task["id"],
                image_path=task["image_name"],
                question=task["question"],
                original_answer=task["answer"],
                parsed=parsed,
            )
            fout.write(json.dumps(mcqa, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Done. {n_ok} succeeded, {n_fail} failed. Output → {args.output}")


def run_api(args):
    """Fallback: use OpenAI-compatible API (Gemini / GPT / etc.)."""
    import asyncio
    import base64
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY", ""),
        base_url=args.api_base_url or "https://api.openai.com/v1",
    )

    print(f"Loading metadata from {args.metadata} ...")
    rows = []
    with open(args.metadata, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    tasks = []
    for row in rows:
        convs = row.get("conversations", [])
        pairs = extract_qa_from_conversations(convs)
        if not pairs:
            continue
        qa = pairs[0]
        img_name = row.get("image", row.get("image_relpath", ""))
        img_path = os.path.join(args.image_dir, img_name) if args.image_dir else img_name
        if not os.path.isfile(img_path):
            continue
        tasks.append({
            "id": row.get("id", img_name),
            "image_path": img_path,
            "image_name": img_name,
            "question": qa["question"],
            "answer": qa["answer"],
        })

    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
    print(f"  {len(tasks)} tasks to process.")

    sem = asyncio.Semaphore(args.concurrency)

    async def process_one(task):
        async with sem:
            with open(task["image_path"], "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode()
            text = PROMPT_TEMPLATE.format(question=task["question"], answer=task["answer"])
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=args.api_model,
                        messages=[
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                {"type": "text", "text": text},
                            ]},
                        ],
                        max_tokens=512,
                        temperature=0.7,
                    )
                    raw = resp.choices[0].message.content
                    parsed = parse_json_response(raw)
                    if parsed:
                        return build_mcqa_row(
                            task["id"], task["image_name"],
                            task["question"], task["answer"], parsed,
                        )
                except Exception as e:
                    if attempt == 2:
                        print(f"  FAIL {task['id']}: {e}")
            return None

    async def main():
        results = await asyncio.gather(*[process_one(t) for t in tasks])
        n_ok = sum(1 for r in results if r is not None)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fout:
            for r in results:
                if r is not None:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Done. {n_ok}/{len(tasks)} succeeded. Output → {args.output}")

    asyncio.run(main())


def main():
    parser = argparse.ArgumentParser(description="Generate MCQA data from CoVT")
    parser.add_argument("--backend", choices=["vllm", "api"], default="vllm")
    parser.add_argument("--model_path", default="google/gemma-4-31B-it")
    parser.add_argument("--metadata", required=True, help="Path to metadata.jsonl")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--tensor_parallel", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="0=all")
    parser.add_argument("--api_model", default="gpt-5", help="API model name")
    parser.add_argument("--api_base_url", default="", help="API base URL override")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    if args.backend == "vllm":
        run_vllm(args)
    else:
        run_api(args)


if __name__ == "__main__":
    main()
