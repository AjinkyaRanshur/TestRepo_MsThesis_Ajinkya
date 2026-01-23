#!/bin/bash
#SBATCH --job-name=pred_net_recon_pc_train_20260123_224803
#SBATCH --output=slurm_jobs/pred_net_recon_pc_train_20260123_224803_%j.out
#SBATCH --error=slurm_jobs/pred_net_recon_pc_train_20260123_224803_%j.err
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
echo "Running 3 experiments with 2 GPUs"
echo "Job ID: $SLURM_JOB_ID"

# Running jobs in batches of 2 (one per GPU)

# Batch 1: Jobs 1-2
CUDA_VISIBLE_DEVICES=0 python main.py --config config_0 --model-name recon_t10_c10_uni_s3336 &
JOB1_PID=$!
echo "Started recon_t10_c10_uni_s3336 on GPU 0 (PID: $JOB1_PID)"
CUDA_VISIBLE_DEVICES=1 python main.py --config config_1 --model-name recon_t10_c10_uni_s1134 &
JOB2_PID=$!
echo "Started recon_t10_c10_uni_s1134 on GPU 1 (PID: $JOB2_PID)"

# Wait for this batch to complete
wait
echo "Batch 1 completed at $(date)"
echo "---"

# Batch 2: Jobs 3-3
CUDA_VISIBLE_DEVICES=0 python main.py --config config_2 --model-name recon_t10_c10_uni_s6833 &
JOB3_PID=$!
echo "Started recon_t10_c10_uni_s6833 on GPU 0 (PID: $JOB3_PID)"

# Wait for this batch to complete
wait
echo "Batch 2 completed at $(date)"
echo "---"

echo "All training jobs completed at $(date)"
echo ""

echo "========================================"
echo "Verifying all models completed successfully..."
echo "========================================"

# Check that all models are marked as completed in registry
python << EOF
from model_tracking import get_tracker
tracker = get_tracker()
models = ['recon_t10_c10_uni_s3336', 'recon_t10_c10_uni_s1134', 'recon_t10_c10_uni_s6833']
incomplete = []
for m in models:
    info = tracker.get_model(m)
    if not info or info.get("status") != "completed":
        incomplete.append(m)
if incomplete:
    print(f"WARNING: {len(incomplete)} models did not complete:")
    for m in incomplete:
        print(f"  - {m}")
    print("Skipping aggregate plots.")
    exit(1)
else:
    print(f"✓ All {len(models)} models completed successfully")
    exit(0)
EOF

# Store exit code from Python script
COMPLETION_CHECK=$?

# Only run post-processing if all models completed
if [ $COMPLETION_CHECK -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Generating aggregate plots for seed groups..."
    echo "========================================"
    python post_training_aggregation.py
    echo "========================================"
    echo "✓ Job complete! Check plots/ directory for results"
    echo "========================================"
else
    echo ""
    echo "⚠ Skipping aggregate plots due to incomplete models"
    echo "Run post_training_aggregation.py manually after models complete"
fi