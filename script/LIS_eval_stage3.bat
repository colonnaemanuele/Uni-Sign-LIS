@echo off
set ckpt_path=out/stage3_finetuning/best_checkpoint.pth

:: RGB-pose setting
python fine_tuning.py ^
   --batch-size 1 ^
   --gradient-accumulation-steps 1 ^
   --epochs 1 ^
   --opt AdamW ^
   --lr 3e-4 ^
   --output_dir out/test ^
   --finetune %ckpt_path% ^
   --dataset LIS ^
   --task SLT ^
   --eval ^
   --rgb_support
