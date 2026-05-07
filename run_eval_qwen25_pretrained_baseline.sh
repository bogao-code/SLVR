#!/bin/bash
# Official Qwen2.5-VL-7B-Instruct (before any MCQA / PinReason finetune).
#
# Examples:
#   bash run_eval_qwen25_pretrained_baseline.sh
#       -> MMVP+V* (same as --bench all), default OUT under results/qwen25_7b_pretrained_baseline_mmvp_vstar
#   BENCH=vstar bash run_eval_qwen25_pretrained_baseline.sh
#       -> V*Bench only -> results/qwen25_7b_pretrained_vstar_only
#   BENCH=mmvp bash run_eval_qwen25_pretrained_baseline.sh
#
# Needs conda env with torch + transformers (default: PYTHON_BIN=nips).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODEL="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
PY="${PYTHON_BIN:-python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
# all | mmvp | vstar
BENCH="${BENCH:-all}"

case "${BENCH}" in
  all)   OUT_DEFAULT="${ROOT}/results/qwen25_7b_pretrained_baseline_mmvp_vstar" ;;
  mmvp)  OUT_DEFAULT="${ROOT}/results/qwen25_7b_pretrained_mmvp_only" ;;
  vstar) OUT_DEFAULT="${ROOT}/results/qwen25_7b_pretrained_vstar_only" ;;
  *) echo "BENCH must be all, mmvp, or vstar"; exit 1 ;;
esac
OUT="${OUT_DIR:-${OUT_DEFAULT}}"

mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

"${PY}" eval_mmvp.py \
  --model_path "${MODEL}" \
  --mode baseline \
  --bench "${BENCH}" \
  --out_dir "${OUT}" \
  2>&1 | tee "${OUT}/run.log"

echo "Done: ${OUT}"
