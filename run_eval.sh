#!/bin/bash
set -e
cd /path/to/pinreason_v2_qwen
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/path/to/pinreason_v2_qwen/out_v2_qwen_typed/best}"
DATASET="${DATASET:-mmvp}"

python eval.py \
  --model_path "$MODEL_PATH" \
  --mode pinreason \
  --dataset "$DATASET" \
  --n_plan 64 \
  --n_patch 16 \
  --n_full_answer 48 \
  --answer_tokens 32
