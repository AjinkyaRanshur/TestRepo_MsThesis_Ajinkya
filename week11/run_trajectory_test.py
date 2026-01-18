# ============================================================
# FILE 1: run_trajectory_test.py
# ============================================================
"""
Standalone script for running trajectory testing
Called by SLURM job
"""

import argparse
import torch
from test_workflow import run_trajectory_test


class TestConfig:
    def __init__(self, timesteps):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 40
        self.timesteps = timesteps
        self.seed = 42
        
        self.gammaset = [[0.33, 0.33, 0.33, 0.33]]
        self.betaset = [[0.33, 0.33, 0.33, 0.33]]
        self.alphaset = [[0.1, 0.1, 0.1, 0.1]]
        
        self.classification_datasetpath = "custom_illusion_dataset"
        self.recon_datasetpath = "custom_illusion_dataset"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True, help="Model names")
    parser.add_argument("--timesteps", type=int, required=True, help="Test timesteps")
    
    args = parser.parse_args()
    
    config = TestConfig(args.timesteps)
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY TESTING (SLURM)")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")
    
    run_trajectory_test(args.models, args.timesteps, config)
    
    print("\n✓ Trajectory testing completed!")
