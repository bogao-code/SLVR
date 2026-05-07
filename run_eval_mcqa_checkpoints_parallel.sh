#!/bin/bash
# Run MMVP + V*Bench in parallel on 5 GPUs, one checkpoint per GPU.
#
# MCQA-finetuned Qwen checkpoints are vanilla VLMs → use --mode baseline (not pinreason).
#
# Usage:
#   bash run_eval_mcqa_checkpoints_parallel.sh
# Override checkpoint list:
#   CKPTS=(/path/a /path/b ...) bash run_eval_mcqa_checkpoints_parallel.sh
#
# Logs: ${OUT_BASE}/logs/<checkpoint-name>.log
# Results per run: ${OUT_BASE}/<checkpoint-name>_baseline_all/

set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MCQA_OUT="${MCQA_ROOT:-${ROOT}/out_mcqa_pretrain}"
OUT_BASE="${OUT_BASE:-${ROOT}/results/mcqa_pretrain_mmvp_vstar}"

if [[ -z "${CKPTS+x}" ]] || [[ "${#CKPTS[@]}" -eq 0 ]]; then
  CKPTS=(
    "${MCQA_OUT}/checkpoint-500"
    "${MCQA_OUT}/checkpoint-750"
    "${MCQA_OUT}/checkpoint-1000"
    "${MCQA_OUT}/checkpoint-1250"
    "${MCQA_OUT}/checkpoint-1256"
  )
fi

mkdir -p "${OUT_BASE}/logs"

# Use GPUs OFFSET, OFFSET+1, … (default OFFSET=0 uses 0–4 for five jobs).
OFFSET="${GPU_OFFSET:-0}"

echo "Writing under: ${OUT_BASE}"
echo "CUDA_VISIBLE_DEVICES slot = OFFSET + slot_index with OFFSET=${OFFSET}"
echo "Checkpoints (${#CKPTS[@]}):"
for c in "${CKPTS[@]}"; do echo "  ${c}"; done

launch_one() {
  local gpu="$1"
  local ckpt="$2"
  local name
  name="$(basename "${ckpt}")"
  local out="${OUT_BASE}/${name}_baseline_all"
  local log="${OUT_BASE}/logs/${name}.log"

  if [[ ! -d "${ckpt}" ]]; then
    echo "ERROR: missing ${ckpt}" >&2
    return 1
  fi

  echo "[CUDA ${gpu}] ${name} → log: ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python eval_mmvp.py \
    --model_path "${ckpt}" \
    --mode baseline \
    --bench all \
    --out_dir "${out}" \
    >"${log}" 2>&1 &
}

i=0
for ckpt in "${CKPTS[@]}"; do
  launch_one "$((OFFSET + i))" "${ckpt}" || exit 1
  i=$((i + 1))
done

wait
echo ""
echo "All jobs finished."
echo "  Summaries: ${OUT_BASE}/*/combined_summary.json"
echo "  Aggregate: python scripts/aggregate_mcqa_eval_summaries.py --root ${OUT_BASE}"
