#!/bin/bash

# ============================================================
# Submit Multiple Experiments in Parallel
# This script submits multiple SLURM jobs, each with a different GPU
# ============================================================

# Array of experiment configurations
EXPERIMENTS=(
    "experiment_configs/reconstruction_experiment.json"
    "experiment_configs/classification_experiment.json"
    "experiment_configs/testing_experiment.json"
)

# GPU allocation per job (if you want to use different GPUs)
GPUS=(0 1 2)

echo "============================================================"
echo "Submitting Multiple Experiments"
echo "============================================================"
echo "Total Experiments: ${#EXPERIMENTS[@]}"
echo ""

# Create logs directory
mkdir -p logs

# Submit jobs
JOB_IDS=()
for i in "${!EXPERIMENTS[@]}"; do
    CONFIG="${EXPERIMENTS[$i]}"
    
    if [ ! -f "$CONFIG" ]; then
        echo "⚠️  Warning: Config file not found: $CONFIG"
        continue
    fi
    
    echo "Submitting experiment $((i+1))/${#EXPERIMENTS[@]}: $CONFIG"
    
    # Submit job and capture job ID
    JOB_ID=$(sbatch --parsable submit_experiment.sh "$CONFIG")
    
    if [ $? -eq 0 ]; then
        JOB_IDS+=($JOB_ID)
        echo "  ✅ Submitted: Job ID $JOB_ID"
    else
        echo "  ❌ Failed to submit"
    fi
    
    echo ""
done

echo "============================================================"
echo "Submission Complete"
echo "============================================================"
echo "Submitted Jobs: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Cancel all jobs: scancel ${JOB_IDS[@]}"
echo "============================================================"

# Save job IDs to file
echo "${JOB_IDS[@]}" > logs/submitted_jobs_$(date +%Y%m%d_%H%M%S).txt
