#!/bin/bash

#SBATCH -A IscrC_EXAM
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pretrained_v2
#SBATCH --out=logs/pretrained_v2.out
#SBATCH --err=logs/pretrained_v2.out

module load spack
spack load cuda@12.2.91 2>/dev/null
spack load ffmpeg@6.0 2>/dev/null
source .venv/bin/activate


echo "Starting Stage 2 Pre-training..."
output_dir=out/stage2_pretraining
ckpt_path=out/stage1_pretraining/best_checkpoint.pth

# Check if stage1 checkpoint exists
if [ ! -f "$ckpt_path" ]; then
    echo "Error: Stage 1 checkpoint not found at $ckpt_path"
    exit 1
fi

srun uv run deepspeed pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 8 \
   --epochs 5 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset LIS \
   --rgb_support

wait 
if [ $? -ne 0 ]; then
    echo "Error: Stage 2 pre-training failed"
    exit 1
fi

echo "Stage 2 completed successfully."