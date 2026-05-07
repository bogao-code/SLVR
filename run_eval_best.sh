#!/bin/bash
set -e
cd /path/to/pinreason_v2_qwen
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/pinreason_v2_qwen/out_v2_qwen_typed}"
STEP="${STEP:-best}"
DATASET="${DATASET:-mmvp}"
N_PLAN="${N_PLAN:-64}"
N_PATCH="${N_PATCH:-16}"
N_FULL_ANSWER="${N_FULL_ANSWER:-8}"
ANSWER_TOKENS="${ANSWER_TOKENS:-8}"
CHOICE_SCORING="${CHOICE_SCORING:-0}"
PRETTY="${PRETTY:-0}"
LIMIT="${LIMIT:-0}"

if [ -z "${MODEL_PATH:-}" ]; then
  if [ "$STEP" = "best" ]; then
    MODEL_PATH="$OUTPUT_ROOT/best"
  else
    MODEL_PATH="$OUTPUT_ROOT/checkpoint-$STEP"
  fi
fi

OUT_DIR="${OUT_DIR:-/path/to/pinreason_v2_qwen/results/pinreason_${DATASET}/step=${STEP}_plan=${N_PLAN}_patch=${N_PATCH}_full=${N_FULL_ANSWER}_typed_lvr}"

python eval.py \
  --model_path "$MODEL_PATH" \
  --mode pinreason \
  --dataset "$DATASET" \
  --out_dir "$OUT_DIR" \
  --n_plan "$N_PLAN" \
  --n_patch "$N_PATCH" \
  --n_full_answer "$N_FULL_ANSWER" \
  --answer_tokens "$ANSWER_TOKENS" \
  --limit "$LIMIT" \
  $( [ "$CHOICE_SCORING" = "1" ] && echo "--choice_scoring" ) \
  $( [ "$PRETTY" = "1" ] && echo "--pretty" )
