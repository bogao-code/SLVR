# SLVR for Qwen2.5-VL

This directory contains the training and evaluation code for PinReason v2 on
Qwen2.5-VL. The method trains typed latent visual reasoning blocks from
Visual-CoT/GQA supervision. The repository is prepared for anonymous review:
no author names, institution names, private keys, or external service tokens are
required by the code.

## Overview

The main training entrypoint is `train.py`. It fine-tunes
`Qwen2.5-VL-7B-Instruct` with four typed latent blocks:

```text
<plan_lvr_start>  <lvr> x K1  <plan_lvr_end>
<bbox_lvr_start>  <lvr> x 3   <bbox_lvr_end>
<patch_lvr_start> <lvr> x K3  <patch_lvr_end>
<full_lvr_start>  <lvr> x K4  <full_lvr_end>
<answer> short_answer </answer>
```

The latent blocks are supervised by:

- plan tokens from the Visual-CoT `thought` field, referred to as `plan` in the
  paper;
- normalized bounding-box coordinates;
- answer-relevant image patch embeddings selected inside the grounding box;
- full-answer token embeddings.

The ordinary language-model cross entropy is applied to non-latent assistant
tokens and the final short answer. The `<lvr>` placeholder tokens are masked
from cross entropy and trained with MSE losses in `src/forward.py`.

## Directory Structure

```text
.
|-- train.py                         # Main PinReason v2 training entrypoint
|-- train_mcqa.py                    # Optional MCQA pretraining entrypoint
|-- eval.py                          # MMVP/V* evaluation wrapper
|-- eval_mmvp.py                     # Decoding and scoring helpers
|-- eval_mmvp_vllm.py                # vLLM baseline utility
|-- run_train.sh                     # Example 8-GPU PinReason training script
|-- run_mcqa_pretrain.sh             # Example MCQA pretraining script
|-- run_eval*.sh                     # Example evaluation scripts
|-- accelerate_config.yaml           # Accelerate distributed config
|-- scripts/
|   |-- aggregate_mcqa_eval_summaries.py
|   |-- download_covt.py
|   `-- generate_mcqa.py
`-- src/
    |-- constants.py                 # Special tokens
    |-- dataset.py                   # Visual-CoT/GQA dataset and collator
    |-- forward.py                   # Qwen2.5-VL forward patch and latent losses
    |-- trainer.py                   # Loss composition
    |-- params.py                    # HF argument dataclasses
    |-- bbox_utils.py                # Box-to-patch mapping utilities
    `-- mcqa_dataset.py              # Optional MCQA dataset
```

Generated directories such as `out_*`, `results`, `__pycache__`, and checkpoint
files are intentionally excluded from the anonymous code package.

## Environment

The code was developed with Python 3.11 and PyTorch on NVIDIA GPUs. Install the
core dependencies with:

```bash
pip install -r requirements.txt
```

For faster training, install a FlashAttention build compatible with your CUDA
and PyTorch versions. If FlashAttention is unavailable, add
`--disable_flash_attn2 true` or use the fallback already built into the training
script.

## Data

The main experiment uses the GQA split of Visual-CoT. Each JSONL row should
contain:

```json
{
  "question": "Who is wearing a shirt?",
  "answer": "girl",
  "full_answer": "The girl is wearing a shirt.",
  "image": "2331819.jpg",
  "width": 500,
  "height": 375,
  "bboxs": [[214, 0, 433, 374]],
  "dataset": "gqa",
  "split": "train",
  "thought": "1. We are looking for someone who is wearing a shirt in the image. 2. First, let's identify all the people in the image. 3. Next, we need to determine who among the people is wearing a shirt. 4. After identifying the person, we can state the answer based on the observation from the image."
}
```

In the paper, the `thought` field is called `plan`. The code keeps the original
field name because it follows the released Visual-CoT metadata.

Expected paths in the provided scripts are examples only:

```text
/path/to/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl
/path/to/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl
/path/to/Visual-CoT/cot_extracted/cot_image_data/gqa
```

Update `run_train.sh` or pass the paths directly on the command line for your
environment.

## Training

Example command:

```bash
cd /path/to/pinreason_v2_qwen
accelerate launch --config_file accelerate_config.yaml train.py \
  --model_id /path/to/Qwen2.5-VL-7B-Instruct \
  --train_data /path/to/gqa_cot_train.jsonl \
  --eval_data /path/to/gqa_cot_val.jsonl \
  --image_folder /path/to/gqa/images \
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
  --eval_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --gradient_checkpointing true \
  --freeze_vision true \
  --ddp_find_unused_parameters false \
  --report_to tensorboard
```

`run_train.sh` contains the same configuration with local paths. To resume from
a checkpoint:

```bash
RESUME_FROM=/path/to/checkpoint bash run_train.sh
```

## Evaluation

The generic evaluation wrapper supports MMVP and V*:

```bash
python eval.py \
  --model_path ./out_v2_qwen_typed/best \
  --mode pinreason \
  --dataset all \
  --n_plan 64 \
  --n_patch 16 \
  --n_full_answer 8 \
  --answer_tokens 32 \
  --out_dir ./results/pinreason_eval
```

For a Qwen2.5-VL baseline without latent blocks:

```bash
python eval.py \
  --model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --mode baseline \
  --dataset all \
  --answer_tokens 32
```

The default MMVP and V* paths in `eval.py` are local examples. Override
`--data_path` and `--image_root` when evaluating a single dataset.

## Anonymous Submission Checklist

Before submitting the code archive:

- include source files, scripts, `README.md`, `requirements.txt`, and
  `accelerate_config.yaml`;
- exclude checkpoints, generated results, caches, logs, and local datasets;
- replace absolute local paths in shell scripts with `/path/to/...` if the
  submission system exposes scripts verbatim;
- do not include pretrained model weights unless explicitly allowed by the
  venue.

