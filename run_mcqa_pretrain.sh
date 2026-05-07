#!/bin/bash
set -e
cd /path/to/pinreason_v2_qwen
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
  --config_file accelerate_config.yaml \
  train_mcqa.py \
  --model_id /path/to/Qwen2.5-VL-7B-Instruct \
  --mcqa_data /path/to/covt_part6/mcqa.jsonl \
  --eval_data /path/to/covt_part6/mcqa_val_50.jsonl \
  --image_folder /path/to/covt_part6/images \
  --max_seq_length 2048 \
  --max_pixels 401408 \
  --output_dir ./out_mcqa_pretrain_qwen \
  --bf16 true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 5 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 500 \
  --save_steps 500 \
  --gradient_checkpointing true \
  --dataloader_num_workers 8 \
  --ddp_find_unused_parameters false \
  --report_to tensorboard
