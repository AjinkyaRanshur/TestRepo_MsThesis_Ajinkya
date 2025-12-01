# Batch Experiment System - Complete Guide

## Overview

This system converts your interactive experiment interface into a batch processing system for SLURM clusters. It allows you to:
- Submit experiments as SLURM jobs
- Run multiple tasks sequentially on a single GPU
- Track progress with detailed logging
- Submit multiple experiments in parallel (different GPUs)

## File Structure

```
week10/
├── batch_runner.py              # Main batch execution script
├── submit_experiment.sh         # SLURM submission script
├── submit_multiple.sh           # Submit multiple jobs in parallel
├── monitor_progress.sh          # Real-time progress monitoring
├── experiment_configs/          # Experiment configuration files
│   ├── reconstruction_experiment.json
│   ├── classification_experiment.json
│   └── testing_experiment.json
├── experiment_logs/             # Progress tracking files (auto-created)
├── logs/                        # SLURM output logs (auto-created)
└── result_folder/               # Experiment results
```

## Setup Instructions

### 1. Create Directory Structure

```bash
cd ~/TestRepo_MsThesis_Ajinkya/week10

# Create necessary directories
mkdir -p experiment_configs
mkdir -p logs
mkdir -p experiment_logs

# Make scripts executable
chmod +x submit_experiment.sh
chmod +x submit_multiple.sh
chmod +x monitor_progress.sh
```

### 2. Create Experiment Configuration Files

Create JSON configuration files in the `experiment_configs/` directory. See examples below.

### 3. Update Paths

Edit `submit_experiment.sh` and update:
- Line 42: `cd /home/ajinkyar/TestRepo_MsThesis_Ajinkya/week10` → your path
- Line 10: Your email address for notifications

## Experiment Configuration Format

### Reconstruction Training

```json
{
  "experiment_type": "reconstruction",
  "patterns": [
    "Uniform",
    "Gamma Increasing",
    "Gamma Decreasing"
  ],
  "recon_timesteps": 10,
  "iterations": 20,
  "datasetpath": "/home/ajinkyar/datasets/",
  "log_dir": "experiment_logs/reconstruction"
}
```

### Classification Training

```json
{
  "experiment_type": "classification",
  "patterns": ["Uniform", "Beta Increasing"],
  "recon_timesteps": 10,
  "class_timesteps": 10,
  "iterations": 25,
  "datasetpath": "data/visual_illusion_dataset",
  "log_dir": "experiment_logs/classification"
}
```

### Testing

```json
{
  "experiment_type": "testing",
  "trained_pattern": "Uniform",
  "model_config": {
    "recon": 10,
    "class": 10
  },
  "test_patterns": ["Uniform", "Gamma Increasing"],
  "test_timesteps": 10,
  "log_dir": "experiment_logs/testing"
}
```

## Usage

### Option 1: Submit Single Experiment

```bash
# Submit a single experiment
sbatch submit_experiment.sh experiment_configs/reconstruction_experiment.json

# Check job status
squeue -u $USER

# View live output
tail -f logs/experiment_<JOB_ID>.out
```

### Option 2: Submit Multiple Experiments in Parallel

```bash
# Edit submit_multiple.sh to list your experiments
nano submit_multiple.sh

# Submit all experiments
./submit_multiple.sh

# This will submit multiple jobs, each using a different GPU
```

### Option 3: Monitor Progress

```bash
# Start the progress monitor (updates every 30 seconds)
./monitor_progress.sh

# Or view a specific log directory
./monitor_progress.sh experiment_logs/reconstruction
```

## Progress Tracking

The system creates two types of log files:

### 1. JSON Log (experiment_log_TIMESTAMP.json)

Detailed machine-readable log with all task information:
```json
{
  "start_time": "20250129_143022",
  "total_tasks": 6,
  "completed_tasks": 4,
  "failed_tasks": 1,
  "tasks": [
    {
      "name": "Recon_Uniform",
      "type": "reconstruction",
      "status": "completed",
      "duration": 1234.5,
      ...
    }
  ]
}
```

### 2. Progress Text File (progress_TIMESTAMP.txt)

Human-readable progress summary:
```
Experiment Progress
==================================================
Total Tasks: 6
Completed: 4/6 (66.7%)
Failed: 1
Remaining: 1

Last Updated: 2025-01-29 14:45:23
==================================================

✅ [1/6] Recon_Uniform - completed
    Duration: 1234.5s
✅ [2/6] Recon_Gamma_Increasing - completed
    Duration: 1189.2s
❌ [3/6] Recon_Beta_Increasing - failed
    Error: CUDA out of memory
🔄 [4/6] Recon_Beta_Decreasing - running
⏳ [5/6] Recon_Custom - pending
```

## Example Workflows

### Workflow 1: Train All Patterns (Reconstruction)

```bash
# 1. Create config
cat > experiment_configs/train_all_recon.json << EOF
{
  "experiment_type": "reconstruction",
  "patterns": [
    "Uniform",
    "Gamma Increasing",
    "Gamma Decreasing",
    "Beta Increasing",
    "Beta Decreasing",
    "Beta Inc & Gamma Dec"
  ],
  "recon_timesteps": 10,
  "iterations": 20,
  "datasetpath": "/home/ajinkyar/datasets/",
  "log_dir": "experiment_logs/recon_all"
}
EOF

# 2. Submit job
sbatch submit_experiment.sh experiment_configs/train_all_recon.json

# 3. Monitor
./monitor_progress.sh experiment_logs/recon_all
```

### Workflow 2: Hyperparameter Sweep

```bash
# Create multiple configs with different timesteps
for t in 1 5 10 20; do
  cat > experiment_configs/recon_t${t}.json << EOF
{
  "experiment_type": "reconstruction",
  "patterns": ["Uniform"],
  "recon_timesteps": ${t},
  "iterations": 20,
  "datasetpath": "/home/ajinkyar/datasets/",
  "log_dir": "experiment_logs/recon_t${t}"
}
EOF
done

# Submit all in parallel
for t in 1 5 10 20; do
  sbatch submit_experiment.sh experiment_configs/recon_t${t}.json
done
```

### Workflow 3: Training Pipeline

```bash
# 1. Train reconstruction
sbatch submit_experiment.sh experiment_configs/reconstruction_experiment.json

# Wait for completion, then:

# 2. Train classification
sbatch submit_experiment.sh experiment_configs/classification_experiment.json

# 3. Run testing
sbatch submit_experiment.sh experiment_configs/testing_experiment.json
```

## Useful SLURM Commands

```bash
# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# View job output (live)
tail -f logs/experiment_<JOB_ID>.out

# View job errors
tail -f logs/experiment_<JOB_ID>.err

# Check past jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS

# Check GPU usage
squeue -u $USER -o "%.18i %.9P %.8T %.4D %.14C %.10m %b"
```

## Debugging

### If a job fails immediately:

```bash
# Check SLURM error log
cat logs/experiment_<JOB_ID>.err

# Common issues:
# 1. Wrong conda environment name
# 2. Missing directories
# 3. Wrong file paths
# 4. Permission issues
```

### If a task fails during execution:

```bash
# Check the JSON log for error details
cat experiment_logs/<log_dir>/experiment_log_*.json | jq '.tasks[] | select(.status=="failed")'

# Check the progress file
cat experiment_logs/<log_dir>/progress_*.txt
```

### Test locally before submitting:

```bash
# Run locally first (no SLURM)
python batch_runner.py --config experiment_configs/test_config.json
```

## Resource Management

### Adjust SLURM Resources

Edit `submit_experiment.sh`:

```bash
#SBATCH --time=48:00:00      # Max time (increase if needed)
#SBATCH --mem=32G            # Memory (increase if OOM errors)
#SBATCH --cpus-per-task=4    # CPU cores
#SBATCH --gres=gpu:1         # GPUs (usually keep at 1)
```

### For Long Experiments

If experiments take longer than 48 hours:
1. Split into smaller batches
2. Or request longer time: `#SBATCH --time=7-00:00:00` (7 days)

## Advanced Features

### Resume Failed Tasks

The progress tracker saves all task information. You can manually create a new config with only failed tasks:

```python
# Script to extract failed tasks
import json

with open('experiment_logs/reconstruction/experiment_log_*.json') as f:
    data = json.load(f)

failed = [t['params']['pattern'] for t in data['tasks'] if t['status'] == 'failed']
print(failed)
```

### Parallel Jobs on Multiple GPUs

Edit `submit_multiple.sh` to use different GPUs:

```bash
EXPERIMENTS=(
    "experiment_configs/exp1.json"
    "experiment_configs/exp2.json"
)

# Each will get its own GPU automatically
```

## Tips and Best Practices

1. **Start Small**: Test with 1-2 patterns before running all
2. **Monitor Early**: Check first task completes before leaving
3. **Check Logs**: Always verify experiment_logs after completion
4. **Save Configs**: Keep your JSON configs in version control
5. **Resource Estimation**: 
   - Reconstruction: ~30-60 min per pattern
   - Classification: ~2-4 hours per pattern
   - Testing: ~5-10 min per pattern

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Job won't start | Check `squeue -u $USER` - might be queued |
| CUDA out of memory | Reduce batch size in config file |
| File not found | Check all paths are absolute |
| Permission denied | Run `chmod +x *.sh` |
| Import errors | Verify conda environment is correct |
| Task stuck | Check GPU usage: `nvidia-smi` on the node |

## Support

For issues:
1. Check SLURM logs: `logs/experiment_<JOB_ID>.err`
2. Check progress: `experiment_logs/*/progress_*.txt`
3. Check JSON logs: `experiment_logs/*/experiment_log_*.json`
4. Test locally first: `python batch_runner.py --config <config>`
