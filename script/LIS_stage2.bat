
@echo off
set output_dir=out\stage2_pretraining

set ckpt_path=out\stage1_pretraining\best_checkpoint.pth

python pre_training.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 5 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 1 \
   --output_dir %output_dir% \
   --finetune %ckpt_path% \
   --dataset LIS \
   --rgb_support

