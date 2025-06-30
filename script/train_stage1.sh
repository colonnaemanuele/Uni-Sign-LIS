output_dir=out/stage1_pretraining

uv run pre_training.py \
   --batch-size 16 \
   --gradient-accumulation-steps 8 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset LIS
