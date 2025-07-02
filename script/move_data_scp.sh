#!/bin/bash
#SBATCH --job-name=move_data_scp
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=IscrC_EXAM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/move_data_scp_%j.out
#SBATCH --error=logs/move_data_scp_%j.err

# Alternative script using SCP for smaller datasets

module load profile/base
module load openssh

mkdir -p logs

LOG_FILE="logs/move_data_scp_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Configuration
SOURCE_HOST="remote_server"
SOURCE_USER="username"
SOURCE_FILES=(
    "/remote/path/file1.mp4"
    "/remote/path/file2.pkl"
    "/remote/path/data/*.json"
)
DEST_PATH="/leonardo_scratch/fast/IscrC_EXAM/emanuele/Uni-Sign-LIS/dataset/LIS"
SSH_KEY_PATH="$HOME/.ssh/id_rsa"

log_message "Starting SCP transfer job"

mkdir -p "$DEST_PATH"

# Transfer each file/pattern
for source_file in "${SOURCE_FILES[@]}"; do
    log_message "Transferring: $source_file"
    
    scp -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=30 \
        -r "$SOURCE_USER@$SOURCE_HOST:$source_file" \
        "$DEST_PATH/" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log_message "Successfully transferred: $source_file"
    else
        log_message "Failed to transfer: $source_file"
    fi
done

log_message "SCP transfer job completed"
