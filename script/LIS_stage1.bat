@echo off
set output_dir=out\stage1_pretraining

python pre_training.py ^
   --batch-size 1 ^
   --gradient-accumulation-steps 1 ^
   --epochs 1 ^
   --opt AdamW ^
   --lr 3e-4 ^
   --quick_break 1 ^
   --output_dir %output_dir% ^
   --dataset LIS 
