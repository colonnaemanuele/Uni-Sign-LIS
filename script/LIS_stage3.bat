@echo off
set output_dir=out/stage3_finetuning

:: RGB-pose setting
set ckpt_path=out/stage2_pretraining/best_checkpoint.pth

python fine_tuning.py ^
  --batch-size 1 ^
  --gradient-accumulation-steps 1 ^
  --epochs 1 ^
  --opt AdamW ^
  --lr 3e-4 ^
  --output_dir %output_dir% ^
  --finetune %ckpt_path% ^
  --dataset LIS ^
  --task SLT ^
  --rgb_support 




