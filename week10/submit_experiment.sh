#!/bin/bash
#SBATCH --job-name=pred_net_exp
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ranshur.ajinkya@students.iiserpune.ac.in

# ============================================================
# SLURM Job Script for Batch Experiments
# Usage: sbatch submit_experiment.sh <config_file>
# Example: sbatch submit_experiment.sh configs/reconstruction_experiment.json
# ============================================================

# Print job information
echo "============================================================"
echo "Job Information"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "============================================================"
echo ""

# Get config file from command line argument
CONFIG_FILE=${1:-"experiment_configs/reconstruction_experiment.json"}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "Configuration File: $CONFIG_FILE"
echo ""

# Load conda environment
echo "Loading conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cuda_pyt

# Verify CUDA availability
echo "============================================================"
echo "GPU Information"
echo "============================================================"
nvidia-smi
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p logs
mkdir -p experiment_logs
mkdir -p result_folder/{trajectories,bar_plots,heatmaps,summaries}
mkdir -p models

# Navigate to project directory
cd /home/ajinkyar/TestRepo_MsThesis_Ajinkya/week10

# Run the batch experiment
echo "============================================================"
echo "Starting Batch Experiment"
echo "============================================================"
python batch_runner.py --config "$CONFIG_FILE"

EXIT_CODE=$?

# Print completion information
echo ""
echo "============================================================"
echo "Job Complete"
echo "============================================================"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

# Send notification (optional)
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
else
    echo "❌ Experiment failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
