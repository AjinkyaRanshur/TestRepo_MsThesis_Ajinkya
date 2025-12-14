"""
Simple Model Tracking and Batch Submission System
Keep it simple, keep it useful.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import subprocess

def create_slurm_script(base_config):
    
    config_paths=create_config_files(base_config.seeds,base_config.patterns,base_config.train_cond,base_config.epochs,base_config.lr_list,base_config.timesteps,base_config.model_name,last_neurons)
                
    """ CREATING SLURM SCRIPT"""
    os.makedirs(output_dir, exist_ok=True)
    
    script = f"""#!/bin/bash
	#SBATCH --job-name={job_name}
	#SBATCH --output={output_dir}/{job_name}_%j.out
	#SBATCH --error={output_dir}/{job_name}_%j.err
	#SBATCH --ntasks={len(config_paths)}
	#SBATCH --cpus-per-task=4
	#SBATCH --gres=gpu:2
	#SBATCH --time=24:00:00

	# Load your environment
	source ~/.bashrc
	conda activate cuda_pyt

	# Run training
        for cfg in configs/generated/*.py;do
		python main.py --config "$cfg" &
	done
	wait

	echo "Job finished at $(date)"
	"""
    
    script_path = f"{output_dir}/{job_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    return script_path
    


def submit_sbatch(script_path):

	result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"✓ Submitted {job_name} (Job ID: {job_id})")
            submitted.append((job_name, job_id))
            tracker.update_status(job_name, "running")
        else:
            print(f"✗ Failed to submit {job_name}")
            print(f"  Error: {result.stderr}")
    
    return submitted





