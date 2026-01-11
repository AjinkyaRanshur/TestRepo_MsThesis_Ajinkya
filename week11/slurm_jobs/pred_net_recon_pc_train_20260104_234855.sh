#!/bin/bash
#SBATCH --job-name=pred_net_recon_pc_train_20260104_234855
#SBATCH --output=slurm_jobs/pred_net_recon_pc_train_20260104_234855_%j.out
#SBATCH --error=slurm_jobs/pred_net_recon_pc_train_20260104_234855_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G

# Load your environment
source ~/.bashrc
conda activate cuda_pyt

# Print job info
echo "Job started at $(date)"
echo "Running 3 experiments"
echo "Job ID: $SLURM_JOB_ID"

# Run training jobs in parallel
# Experiment 1: pc_recon10_custom_illusion_dataset_Uniform_seed42
CUDA_VISIBLE_DEVICES=0 python main.py --config config_0 --model-name pc_recon10_custom_illusion_dataset_Uniform_seed42 &
echo "Started job 1 (Model: pc_recon10_custom_illusion_dataset_Uniform_seed42) on GPU 0"

# Experiment 2: pc_recon10_cifar10_Uniform_seed42
CUDA_VISIBLE_DEVICES=1 python main.py --config config_1 --model-name pc_recon10_cifar10_Uniform_seed42 &
echo "Started job 2 (Model: pc_recon10_cifar10_Uniform_seed42) on GPU 1"

# Experiment 3: pc_recon10_stl10_Uniform_seed42
CUDA_VISIBLE_DEVICES=0 python main.py --config config_2 --model-name pc_recon10_stl10_Uniform_seed42 &
echo "Started job 3 (Model: pc_recon10_stl10_Uniform_seed42) on GPU 0"

# Wait for all jobs to complete
wait

echo "All jobs completed at $(date)"