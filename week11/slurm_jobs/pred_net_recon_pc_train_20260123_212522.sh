#!/bin/bash
#SBATCH --job-name=pred_net_recon_pc_train_20260123_212522
#SBATCH --output=slurm_jobs/pred_net_recon_pc_train_20260123_212522_%j.out
#SBATCH --error=slurm_jobs/pred_net_recon_pc_train_20260123_212522_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G

# Load environment
source ~/.bashrc
conda activate cuda_pyt

# Print job info
echo "Job started at $(date)"
echo "Running 3 experiments in parallel"
echo "Job ID: $SLURM_JOB_ID"

# Run all jobs in parallel, each on its own GPU

# Job 1: recon_t10_c10_uni_s8723
CUDA_VISIBLE_DEVICES=0 python main.py --config config_0 --model-name recon_t10_c10_uni_s8723 &
JOB1_PID=$!
echo "Started recon_t10_c10_uni_s8723 on GPU 0 (PID: $JOB1_PID)"

# Job 2: recon_t10_c10_uni_s2535
CUDA_VISIBLE_DEVICES=1 python main.py --config config_1 --model-name recon_t10_c10_uni_s2535 &
JOB2_PID=$!
echo "Started recon_t10_c10_uni_s2535 on GPU 1 (PID: $JOB2_PID)"

# Job 3: recon_t10_c10_uni_s4214
CUDA_VISIBLE_DEVICES=2 python main.py --config config_2 --model-name recon_t10_c10_uni_s4214 &
JOB3_PID=$!
echo "Started recon_t10_c10_uni_s4214 on GPU 2 (PID: $JOB3_PID)"

# Wait for all jobs to complete
wait
echo "All training jobs completed at $(date)"
echo "---"

echo ""
echo "========================================"
echo "Verifying all models completed successfully..."
echo "========================================"

# ✅ Check that all models are marked as completed in registry
python << EOF
from model_tracking import get_tracker
tracker = get_tracker()
models = ['recon_t10_c10_uni_s8723', 'recon_t10_c10_uni_s2535', 'recon_t10_c10_uni_s4214']
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