"""
Simple Model Tracking and Batch Submission System
Keep it simple, keep it useful.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
from create_config import create_config_files
from model_tracking import get_tracker


def create_slurm_script(base_config, output_dir="slurm_jobs"):
    """
    Create SLURM batch submission script for multiple experiments
    
    Args:
        base_config: Dictionary containing experiment parameters
        output_dir: Directory to store SLURM scripts and logs
    
    Returns:
        Path to created SLURM script and list of model names
    """
    
  
    
    # Create config files and register models
    if base_config["train_cond"] == "classification_training_shapes":
        config_paths, model_names = create_config_files(
            seeds=base_config["seeds"],
            patterns=base_config["selected_patterns"],
            train_cond=base_config["train_cond"],
            epochs=base_config["epochs"],
            lr_list=base_config["lr"],
            timesteps=base_config["timesteps"],
            number_of_classes=base_config["number_of_classes"],
            base_recon_models=base_config.get("base_recon_models", []),
            checkpoint_epochs=base_config.get("checkpoint_epochs", []),
            dataset_list=base_config["dataset_list"]
        )
    else:
        config_paths, model_names = create_config_files(
            seeds=base_config["seeds"],
            patterns=base_config["selected_patterns"],
            train_cond=base_config["train_cond"],
            epochs=base_config["epochs"],
            lr_list=base_config["lr"],
            timesteps=base_config["timesteps"],
            number_of_classes=[10],
            dataset_list=base_config["dataset_list"]
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"pred_net_{base_config['train_cond']}_{timestamp}"
    
    # Calculate resources
    num_jobs = len(config_paths)
    gpus_needed = min(2, (num_jobs + 1) // 2)  # 2 GPUs max, 1 GPU per 2 jobs
    
    # Build script line by line to avoid indentation issues
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_dir}/{job_name}_%j.out",
        f"#SBATCH --error={output_dir}/{job_name}_%j.err",
        f"#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task=4",
        f"#SBATCH --gres=gpu:{gpus_needed}",
        f"#SBATCH --partition=gpu",
        f"#SBATCH --time=4-00:00:00",
        f"#SBATCH --mem=32G",
        "",
        "# Load your environment",
        "source ~/.bashrc",
        "conda activate cuda_pyt",
        "",
        "# Print job info",
        'echo "Job started at $(date)"',
        f'echo "Running {num_jobs} experiments"',
        'echo "Job ID: $SLURM_JOB_ID"',
        "",
        "# Run training jobs in parallel",
    ]
    
    # Add parallel job execution
    for i, (cfg_path, model_name) in enumerate(zip(config_paths, model_names)):
        gpu_id = i % gpus_needed
        script_lines.extend([
            f"# Experiment {i+1}: {model_name}",
            f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py --config {cfg_path} --model-name {model_name} &",
            f'echo "Started job {i+1} (Model: {model_name}) on GPU {gpu_id}"',
            ""
        ])
    
    # Add wait and completion
    script_lines.extend([
        "# Wait for all jobs to complete",
        "wait",
        "",
        'echo "All jobs completed at $(date)"'
    ])
    
    # Join all lines with newlines
    script = "\n".join(script_lines)
    
    # Write script to file
    script_path = f"{output_dir}/{job_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"\n✓ Created SLURM script: {script_path}")
    print(f"✓ Generated {num_jobs} config files")
    print(f"✓ Registered {len(model_names)} models in tracker")
    
    return script_path, model_names


def submit_sbatch(script_path, model_names):
    """
    Submit SLURM batch job
    
    Args:
        script_path: Path to SLURM script
        model_names: List of model names to update status
    
    Returns:
        List of (job_name, job_id) tuples
    """
    submitted = []
    tracker = get_tracker()
    
    try:
        # Submit the job
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Extract job ID from output (format: "Submitted batch job 12345")
            job_id = result.stdout.strip().split()[-1]
            job_name = os.path.basename(script_path).replace('.sh', '')
            
            print(f"\n✓ Submitted {job_name} (Job ID: {job_id})")
            submitted.append((job_name, job_id))
            
            # Update all model statuses to 'submitted'
            for model_name in model_names:
                tracker.update_status(model_name, "submitted")
            
            # Save submission info
            submission_info = {
                "slurm_job_id": job_id,
                "script_path": script_path,
                "model_names": model_names,
                "submitted_at": datetime.now().isoformat()
            }
            
            info_path = script_path.replace('.sh', '_submission.json')
            with open(info_path, 'w') as f:
                json.dump(submission_info, f, indent=2)
            
            print(f"✓ Submission info saved to {info_path}")
            
        else:
            print(f"\n✗ Failed to submit {script_path}")
            print(f"  Error: {result.stderr}")
    
    except FileNotFoundError:
        print("\n✗ 'sbatch' command not found. Are you on a SLURM cluster?")
        print("  To test locally, you can run the script directly:")
        print(f"  bash {script_path}")
    
    return submitted


def check_job_status(job_id):
    """Check status of a SLURM job"""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1]  # Return status (RUNNING, PENDING, etc.)
        return "UNKNOWN"
    
    except FileNotFoundError:
        return "NO_SLURM"


def monitor_jobs(submission_json_path):
    """Monitor status of submitted jobs"""
    with open(submission_json_path, 'r') as f:
        info = json.load(f)
    
    job_id = info["slurm_job_id"]
    model_names = info["model_names"]
    
    print(f"\nMonitoring Job {job_id}")
    print(f"Models: {len(model_names)}")
    
    status = check_job_status(job_id)
    print(f"Status: {status}")
    
    if status == "COMPLETED":
        tracker = get_tracker()
        for model_name in model_names:
            tracker.update_status(model_name, "completed")
        print("✓ Updated all model statuses to 'completed'")


#if __name__ == "__main__":
    # Example usage
#    example_config = {
#        "train_cond": "recon_pc_train",
#        "epochs": [200],
#        "batch_size": [40],
#        "lr": [0.00005],
#        "timesteps": [10],
#        "number_of_classes": 10,
#       "selected_patterns": ["Uniform", "Gamma Increasing"],
#        "seeds": [42, 123]
#    }
    
#    script_path, model_names = create_slurm_script(example_config)
#    print(f"\nTo submit: sbatch {script_path}")
#    print("Or run: python batch_submissions.py")
