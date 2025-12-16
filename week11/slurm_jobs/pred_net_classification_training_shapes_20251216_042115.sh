#!/bin/bash
#SBATCH --job-name=pred_net_classification_training_shapes_20251216_042115
#SBATCH --output=slurm_jobs/pred_net_classification_training_shapes_20251216_042115_%j.out
#SBATCH --error=slurm_jobs/pred_net_classification_training_shapes_20251216_042115_%j.err
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
# Experiment 1: class_pc_recon10_Uniform_seed42_chk15_Uniform_seed42
CUDA_VISIBLE_DEVICES=0 python main.py --config config_0 --model-name class_pc_recon10_Uniform_seed42_chk15_Uniform_seed42 &
echo "Started job 1 (Model: class_pc_recon10_Uniform_seed42_chk15_Uniform_seed42) on GPU 0"

# Wait for all jobs to complete
wait

echo "All jobs completed at $(date)"