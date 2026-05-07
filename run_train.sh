#!/bin/bash
set -e
cd /path/to/pinreason_v2_qwen
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM")
fi

accelerate launch \
  --config_file accelerate_config.yaml \
  train.py \
  --model_id /path/to/Qwen2.5-VL-7B-Instruct \
  --train_data /path/to/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl \
  --eval_data  /path/to/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl \
  --image_folder /path/to/Visual-CoT/cot_extracted/cot_image_data/gqa \
  --max_seq_length 4096 \
  --max_pixels 401408 \
  --max_plan_tokens 256 \
  --max_full_answer_tokens 48 \
  --n_patch_tokens 0 \
  --patch_fraction 0.5 \
  --patch_cos_threshold 0.5 \
  --min_patches 5 \
  --output_dir ./out_v2_qwen_typed \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 5 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --custom_warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --gradient_checkpointing true \
  --dataloader_num_workers 8 \
  --loss_plan_lambda 0.1 \
  --loss_bbox_lambda 1.0 \
  --loss_patch_lambda 1.0 \
  --loss_answer_lambda 0.5 \
  --loss_answer_open_lambda 2.0 \
  --loss_short_answer_lambda 3.0 \
  --log_loss_components false \
  --ddp_find_unused_parameters false \
  --report_to tensorboard \
  "${RESUME_ARGS[@]}"
