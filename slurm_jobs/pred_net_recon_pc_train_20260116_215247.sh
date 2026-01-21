#!/bin/bash
#SBATCH --job-name=pred_net_recon_pc_train_20260116_215247
#SBATCH --output=slurm_jobs/pred_net_recon_pc_train_20260116_215247_%j.out
#SBATCH --error=slurm_jobs/pred_net_recon_pc_train_20260116_215247_%j.err
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
echo "Running 12 experiments (2 at a time)"
echo "Job ID: $SLURM_JOB_ID"

# FIXED: Run only 2 jobs in parallel at a time
# This prevents registry corruption from simultaneous writes

# Batch 1 - Job 1: recon_t10_ill_uni_s1825
CUDA_VISIBLE_DEVICES=0 python main.py --config config_0 --model-name recon_t10_ill_uni_s1825 &
JOB1_PID=$!
echo "Started recon_t10_ill_uni_s1825 on GPU 0 (PID: $JOB1_PID)"

# Batch 1 - Job 2: recon_t10_c10_uni_s1825
CUDA_VISIBLE_DEVICES=1 python main.py --config config_1 --model-name recon_t10_c10_uni_s1825 &
JOB2_PID=$!
echo "Started recon_t10_c10_uni_s1825 on GPU 1 (PID: $JOB2_PID)"

# Wait for current batch to complete
wait
echo "Batch 1 completed at $(date)"
echo "---"

# Batch 2 - Job 1: recon_t10_stl_uni_s1825
CUDA_VISIBLE_DEVICES=0 python main.py --config config_2 --model-name recon_t10_stl_uni_s1825 &
JOB1_PID=$!
echo "Started recon_t10_stl_uni_s1825 on GPU 0 (PID: $JOB1_PID)"

# Batch 2 - Job 2: recon_t10_ill_uni_s410
CUDA_VISIBLE_DEVICES=1 python main.py --config config_3 --model-name recon_t10_ill_uni_s410 &
JOB2_PID=$!
echo "Started recon_t10_ill_uni_s410 on GPU 1 (PID: $JOB2_PID)"

# Wait for current batch to complete
wait
echo "Batch 2 completed at $(date)"
echo "---"

# Batch 3 - Job 1: recon_t10_c10_uni_s410
CUDA_VISIBLE_DEVICES=0 python main.py --config config_4 --model-name recon_t10_c10_uni_s410 &
JOB1_PID=$!
echo "Started recon_t10_c10_uni_s410 on GPU 0 (PID: $JOB1_PID)"

# Batch 3 - Job 2: recon_t10_stl_uni_s410
CUDA_VISIBLE_DEVICES=1 python main.py --config config_5 --model-name recon_t10_stl_uni_s410 &
JOB2_PID=$!
echo "Started recon_t10_stl_uni_s410 on GPU 1 (PID: $JOB2_PID)"

# Wait for current batch to complete
wait
echo "Batch 3 completed at $(date)"
echo "---"

# Batch 4 - Job 1: recon_t10_ill_uni_s4507
CUDA_VISIBLE_DEVICES=0 python main.py --config config_6 --model-name recon_t10_ill_uni_s4507 &
JOB1_PID=$!
echo "Started recon_t10_ill_uni_s4507 on GPU 0 (PID: $JOB1_PID)"

# Batch 4 - Job 2: recon_t10_c10_uni_s4507
CUDA_VISIBLE_DEVICES=1 python main.py --config config_7 --model-name recon_t10_c10_uni_s4507 &
JOB2_PID=$!
echo "Started recon_t10_c10_uni_s4507 on GPU 1 (PID: $JOB2_PID)"

# Wait for current batch to complete
wait
echo "Batch 4 completed at $(date)"
echo "---"

# Batch 5 - Job 1: recon_t10_stl_uni_s4507
CUDA_VISIBLE_DEVICES=0 python main.py --config config_8 --model-name recon_t10_stl_uni_s4507 &
JOB1_PID=$!
echo "Started recon_t10_stl_uni_s4507 on GPU 0 (PID: $JOB1_PID)"

# Batch 5 - Job 2: recon_t10_ill_uni_s4013
CUDA_VISIBLE_DEVICES=1 python main.py --config config_9 --model-name recon_t10_ill_uni_s4013 &
JOB2_PID=$!
echo "Started recon_t10_ill_uni_s4013 on GPU 1 (PID: $JOB2_PID)"

# Wait for current batch to complete
wait
echo "Batch 5 completed at $(date)"
echo "---"

# Batch 6 - Job 1: recon_t10_c10_uni_s4013
CUDA_VISIBLE_DEVICES=0 python main.py --config config_10 --model-name recon_t10_c10_uni_s4013 &
JOB1_PID=$!
echo "Started recon_t10_c10_uni_s4013 on GPU 0 (PID: $JOB1_PID)"

# Batch 6 - Job 2: recon_t10_stl_uni_s4013
CUDA_VISIBLE_DEVICES=1 python main.py --config config_11 --model-name recon_t10_stl_uni_s4013 &
JOB2_PID=$!
echo "Started recon_t10_stl_uni_s4013 on GPU 1 (PID: $JOB2_PID)"

# Wait for current batch to complete
wait
echo "Batch 6 completed at $(date)"
echo "---"

echo "All jobs completed at $(date)"