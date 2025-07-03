#!/bin/bash

#SBATCH --account=IscrC_EXAM
#SBATCH --job-name=move_data
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/move_data_%j.out
#SBATCH --error=logs/move_data_%j.err

mkdir -p logs
LOG_FILE="logs/move_data_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

handle_error() {
    log_message "ERROR: $1"
    exit 1
}

log_message "Starting data move job on $(hostname)"
log_message "Job ID: $SLURM_JOB_ID"
log_message "Working directory: $(pwd)"

# Load configuration from .env file
ENV_FILE="$(dirname "$(dirname "$(realpath "$0")")")/.env"
if [ -f "$ENV_FILE" ]; then
    log_message "Loading configuration from $ENV_FILE"
    # Source the .env file
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a  # stop automatically exporting
    log_message "Configuration loaded successfully"
else
    handle_error ".env file not found at $ENV_FILE"
fi

# Expand variables (in case they contain other variables like $HOME)
SSH_KEY_PATH=$(eval echo "$SSH_KEY_PATH")

log_message "Configuration:"
log_message "  - Source: $SOURCE_USER@$SOURCE_HOST:$SOURCE_PATH"
log_message "  - Destination: $DEST_PATH"
log_message "  - SSH Key: $SSH_KEY_PATH"

if [ ! -f "$SSH_KEY_PATH" ]; then
    handle_error "SSH key not found at $SSH_KEY_PATH"
fi

mkdir -p "$DEST_PATH"
log_message "Starting data transfer from $SOURCE_USER@$SOURCE_HOST:$SOURCE_PATH to $DEST_PATH"

rsync $RSYNC_OPTIONS \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o ConnectTimeout=30" \
    "$SOURCE_USER@$SOURCE_HOST:$SOURCE_PATH/" \
    "$DEST_PATH/" 2>&1 | tee -a "$LOG_FILE"

# Check if transfer was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_message "Data transfer completed successfully"    
    log_message "Verifying transferred data..."
    
    FILE_COUNT=$(find "$DEST_PATH" -type f | wc -l)
    
    log_message "Transfer summary:"
    log_message "  - Total files: $FILE_COUNT"
    log_message "  - Destination: $DEST_PATH"
    
else
    handle_error "Data transfer failed with exit code ${PIPESTATUS[0]}"
fi

# Optional: Set permissions
# log_message "Setting appropriate permissions..."
# chmod -R 755 "$DEST_PATH"

log_message "Job completed successfully"
log_message "Log file saved to: $LOG_FILE"