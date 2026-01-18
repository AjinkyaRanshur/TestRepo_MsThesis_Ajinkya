# ============================================================
# FILE 3: run_grid_search_test.py
# ============================================================
"""
Standalone script for running grid search
Called by SLURM job
"""

import argparse
import torch
import numpy as np
from grid_search_testing import run_grid_search, print_grid_search_summary


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
    parser.add_argument("--gamma-start", type=float, required=True)
    parser.add_argument("--gamma-stop", type=float, required=True)
    parser.add_argument("--gamma-step", type=float, required=True)
    parser.add_argument("--beta-start", type=float, required=True)
    parser.add_argument("--beta-stop", type=float, required=True)
    parser.add_argument("--beta-step", type=float, required=True)
    
    args = parser.parse_args()
    
    gamma_range = (args.gamma_start, args.gamma_stop, args.gamma_step)
    beta_range = (args.beta_start, args.beta_stop, args.beta_step)
    
    config = TestConfig(args.timesteps)
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH (SLURM)")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Gamma range: {gamma_range}")
    print(f"Beta range: {beta_range}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")
    
    results = run_grid_search(args.models, args.timesteps, config, gamma_range, beta_range)
    
    gamma_values = np.arange(*gamma_range)
    beta_values = np.arange(*beta_range)
    print_grid_search_summary(results, gamma_values, beta_values)
    
    print("\n✓ Grid search completed!")
