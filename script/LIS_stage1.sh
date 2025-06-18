#!/bin/bash
output_dir="out/stage1_pretraining"

python pre_training.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 10 \
  --opt AdamW \
  --lr 3e-4 \
  --quick_break 2048 \
  --output_dir "$output_dir" \
  --dataset LIS
