"""
SLURM Testing Submission System
Creates and submits batch jobs for testing classification models
"""

import json
import os
from datetime import datetime
from pathlib import Path
import subprocess


def create_slurm_test_script(test_config, output_dir="slurm_jobs"):
    """
    Create SLURM batch submission script for testing
    
    Args:
        test_config: Dict with test configuration
            - test_type: "trajectory", "pattern", or "grid_search"
            - model_names: List of model names (all seeds)
            - test_timesteps: Number of timesteps
            - test_patterns: List of patterns (for pattern testing)
            - grid_params: Grid search parameters (for grid search)
    
    Returns:
        Path to created SLURM script
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_type = test_config["test_type"]
    job_name = f"test_{test_type}_{timestamp}"
    
    # Build script
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_dir}/{job_name}_%j.out",
        f"#SBATCH --error={output_dir}/{job_name}_%j.err",
        f"#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task=4",
        f"#SBATCH --gres=gpu:1",
        f"#SBATCH --partition=gpu",
        f"#SBATCH --time=12:00:00",
        f"#SBATCH --mem=32G",
        "",
        "# Load environment",
        "source ~/.bashrc",
        "conda activate cuda_pyt",
        "",
        "# Print job info",
        'echo "Job started at $(date)"',
        f'echo "Test type: {test_type}"',
        f'echo "Models: {len(test_config["model_names"])}"',
        f'  --dataset {dataset}',  # ADD THIS LINE
        f'echo "Test timesteps: {test_config["test_timesteps"]}"',
        'echo "Job ID: $SLURM_JOB_ID"',
        "",
    ]
   
    # NEW: Determine dataset from first model
    from model_tracking import get_tracker
    tracker = get_tracker()
    first_model = tracker.get_model(test_config["model_names"][0])
    dataset = "custom_illusion_dataset"  # default
    if first_model:
        dataset = first_model['config'].get('Dataset', 'custom_illusion_dataset')
    
    # Add test-specific command
    if test_type == "trajectory":
        script_lines.extend([
            "# Run trajectory testing",
            "python run_test.py \\",
            f'  --mode trajectory \\',
            f'  --models {" ".join(test_config["model_names"])} \\',
            f'  --timesteps {test_config["test_timesteps"]} \\',
            f'  --dataset {dataset}',
            ""
        ])

    elif test_type == "pattern":
        patterns_str = ",".join(test_config["test_patterns"])
        script_lines.extend([
            "# Run pattern testing",
            "python run_test.py \\",
            f'  --mode pattern \\',
            f'  --models {" ".join(test_config["model_names"])} \\',
            f'  --timesteps {test_config["test_timesteps"]} \\',
            f'  --patterns "{patterns_str}" \\',
            f'  --dataset {dataset}',
            ""
        ])

    elif test_type == "grid_search":
        grid_params = test_config["grid_params"]
        gamma_start, gamma_stop, gamma_step = grid_params["gamma_range"]
        beta_start, beta_stop, beta_step = grid_params["beta_range"]

        script_lines.extend([
            "# Run grid search",
            "python run_test.py \\",
            f'  --mode grid \\',
            f'  --models {" ".join(test_config["model_names"])} \\',
            f'  --timesteps {test_config["test_timesteps"]} \\',
            f'  --gamma-start {gamma_start} --gamma-stop {gamma_stop} --gamma-step {gamma_step} \\',
            f'  --beta-start {beta_start} --beta-stop {beta_stop} --beta-step {beta_step} \\',
            f'  --dataset {dataset}',
            ""
        ])
    
    script_lines.extend([
        'echo "Test completed at $(date)"'
    ])
    
    # Write script
    script = "\n".join(script_lines)
    script_path = f"{output_dir}/{job_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    
    print(f"\n✓ Created SLURM test script: {script_path}")
    print(f"✓ Test type: {test_type}")
    print(f"✓ Models: {len(test_config['model_names'])}")
    print(f"✓ Test timesteps: {test_config['test_timesteps']}")
    print(f"✓ Dataset: {dataset}")  # ADD THIS LINE


    if test_type == "pattern":
        print(f"✓ Patterns: {len(test_config['test_patterns'])}")
    elif test_type == "grid_search":
        import numpy as np
        gamma_range = test_config['grid_params']['gamma_range']
        beta_range = test_config['grid_params']['beta_range']
        n_gamma = len(np.arange(*gamma_range))
        n_beta = len(np.arange(*beta_range))
        print(f"✓ Grid points: {n_gamma} x {n_beta} = {n_gamma * n_beta}")
    
    return script_path


def submit_test_job(script_path, test_config):
    """
    Submit SLURM test job
    
    Args:
        script_path: Path to SLURM script
        test_config: Test configuration dict
    
    Returns:
        Job ID or None
    """
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_name = os.path.basename(script_path).replace('.sh', '')
            
            print(f"\n✓ Submitted {job_name} (Job ID: {job_id})")
            
            # Save submission info
            submission_info = {
                "slurm_job_id": job_id,
                "script_path": script_path,
                "test_type": test_config["test_type"],
                "model_names": test_config["model_names"],
                "test_timesteps": test_config["test_timesteps"],
                "submitted_at": datetime.now().isoformat()
            }
            
            if test_config["test_type"] == "pattern":
                submission_info["test_patterns"] = test_config["test_patterns"]
            elif test_config["test_type"] == "grid_search":
                submission_info["grid_params"] = test_config["grid_params"]
            
            info_path = script_path.replace('.sh', '_submission.json')
            with open(info_path, 'w') as f:
                json.dump(submission_info, f, indent=2)
            
            print(f"✓ Submission info saved to {info_path}")
            
            return job_id
        else:
            print(f"\n✗ Failed to submit {script_path}")
            print(f"  Error: {result.stderr}")
            return None
    
    except FileNotFoundError:
        print("\n✗ 'sbatch' command not found. Are you on a SLURM cluster?")
        print("  To test locally, you can run the script directly:")
        print(f"  bash {script_path}")
        return None


def check_test_job_status(job_id):
    """Check status of a SLURM test job"""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1]
        return "UNKNOWN"
    
    except FileNotFoundError:
        return "NO_SLURM"
