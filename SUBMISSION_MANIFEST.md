# Anonymous Submission Manifest

Include these files/directories:

```text
README.md
requirements.txt
accelerate_config.yaml
train.py
train_mcqa.py
eval.py
eval_mmvp.py
eval_mmvp_vllm.py
run_train.sh
run_mcqa_pretrain.sh
run_eval.sh
run_eval_best.sh
run_eval_mcqa_checkpoints_parallel.sh
run_eval_qwen25_pretrained_baseline.sh
scripts/
src/
```

Exclude these files/directories:

```text
__pycache__/
out_mcqa_pretrain/
out_mcqa_pretrain_qwen/
out_v2_qwen_typed/
results/
*.safetensors
*.bin
*.pt
*.pth
checkpoint-*/
local datasets
pretrained model weights
```

Recommended archive command from the parent directory:

```bash
tar --exclude='__pycache__' \
    --exclude='out*' \
    --exclude='results' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    -czf pinreason_v2_qwen_anonymous.tar.gz pinreason_v2_qwen_anonymous
```

