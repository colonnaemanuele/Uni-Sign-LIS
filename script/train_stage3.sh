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
#SBATCH --job-name=finetuning
#SBATCH --out=logs/finetuning.out
#SBATCH --err=logs/finetuning.out

module load spack
spack load cuda@12.2.91 2>/dev/null
spack load ffmpeg@6.0 2>/dev/null
source .venv/bin/activate

echo "Starting Stage 3 Fine-tuning..."
output_dir=out/stage3_finetuning
ckpt_path=out/stage2_pretraining/best_checkpoint.pth

# Check if stage2 checkpoint exists
if [ ! -f "$ckpt_path" ]; then
    echo "Error: Stage 2 checkpoint not found at $ckpt_path"
    exit 1
fi

srun uv run deepspeed fine_tuning.py \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset LIS \
  --task SLT \
  --rgb_support 

echo "Stage 3 completed successfully."

# example of ISLR
# deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
#    --batch-size 8 \
#    --gradient-accumulation-steps 1 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --output_dir $output_dir \
#    --finetune $ckpt_path \
#    --dataset WLASL \
#    --task ISLR \
#    --max_length 64 \
#    --rgb_support # enable RGB-pose setting

## pose only setting
#ckpt_path=out/stage1_pretraining/best_checkpoint.pth
#
#deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
#  --batch-size 8 \
#  --gradient-accumulation-steps 1 \
#  --epochs 20 \
#  --opt AdamW \
#  --lr 3e-4 \
#  --output_dir $output_dir \
#  --finetune $ckpt_path \
#  --dataset CSL_Daily \
#  --task SLT \
##   --rgb_support # enable RGB-pose setting