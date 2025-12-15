#!/bin/bash
#SBATCH --job-name=pred_net_recoc_train_20251215_013145
#SBATCH --output=slurm_jobs/pred_net_recoc_train_20251215_013145_%j.out
#SBATCH --error=slurm_jobs/pred_net_recoc_train_20251215_013145_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=32G

# Load your environment
source ~/.bashrc
conda activate cuda_pyt

# Print job info
echo "Job started at $(date)"
echo "Running 1 experiments"
echo "Job ID: $SLURM_JOB_ID"

# Run training jobs in parallel
# Experiment 1: None
CUDA_VISIBLE_DEVICES=0 python main.py --config config_0 --model-name None &
echo "Started job 1 (Model: None) on GPU 0"

# Wait for all jobs to complete
wait

echo "All jobs completed at $(date)"