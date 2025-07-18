output_dir=out/stage1_pretraining

uv run deepspeed --include localhost:0,1,2 --master_port 29511 pre_training.py \
   --batch-size 8 \
   --gradient-accumulation-steps 8 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset LIS
