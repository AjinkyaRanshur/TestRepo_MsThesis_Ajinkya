#!/bin/bash
#SBATCH --job-name=pred_net_recon_pc_train_20260119_234202
#SBATCH --output=slurm_jobs/pred_net_recon_pc_train_20260119_234202_%j.out
#SBATCH --error=slurm_jobs/pred_net_recon_pc_train_20260119_234202_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G

# Load environment
source ~/.bashrc
conda activate cuda_pyt

# Print job info
echo "Job started at $(date)"
echo "Running 3 experiments (2 at a time)"
echo "Job ID: $SLURM_JOB_ID"

# Run jobs in batches of 2 to prevent registry corruption

# Batch 1 - Job 1: recon_t10_c10_uni_s4783
CUDA_VISIBLE_DEVICES=0 python main.py --config config_0 --model-name recon_t10_c10_uni_s4783 &
JOB1_PID=$!
echo "Started recon_t10_c10_uni_s4783 on GPU 0 (PID: $JOB1_PID)"

# Batch 1 - Job 2: recon_t10_c10_uni_s5963
CUDA_VISIBLE_DEVICES=1 python main.py --config config_1 --model-name recon_t10_c10_uni_s5963 &
JOB2_PID=$!
echo "Started recon_t10_c10_uni_s5963 on GPU 1 (PID: $JOB2_PID)"

# Wait for current batch to complete
wait
echo "Batch 1 completed at $(date)"
echo "---"

# Batch 2 - Job 1: recon_t10_c10_uni_s5603
CUDA_VISIBLE_DEVICES=0 python main.py --config config_2 --model-name recon_t10_c10_uni_s5603 &
JOB1_PID=$!
echo "Started recon_t10_c10_uni_s5603 on GPU 0 (PID: $JOB1_PID)"

# Wait for current batch to complete
wait
echo "Batch 2 completed at $(date)"
echo "---"

echo "All training jobs completed at $(date)"
echo ""
echo "="
echo "Generating aggregate plots for seed groups..."
echo "="

# Run post-processing aggregation script
python post_training_aggregation.py

echo "="
echo "✓ Job complete! Check plots/ directory for results"
echo "="