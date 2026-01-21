# ============================================================
# FILE 2: run_pattern_test.py
# ============================================================
"""
Standalone script for running pattern testing
Called by SLURM job
"""

import argparse
import torch
from pattern_testing import run_pattern_testing, print_pattern_testing_summary


class TestConfig:
    def __init__(self, timesteps,dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 40
        self.timesteps = timesteps
        self.seed = 42
        
        self.gammaset = [[0.33, 0.33, 0.33, 0.33]]
        self.betaset = [[0.33, 0.33, 0.33, 0.33]]
        self.alphaset = [[0.1, 0.1, 0.1, 0.1]]
        
        self.classification_datasetpath = dataset
        self.recon_datasetpath = dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True, help="Model names")
    parser.add_argument("--timesteps", type=int, required=True, help="Test timesteps")
    parser.add_argument("--patterns", type=str, required=True, help="Comma-separated pattern names")
    parser.add_argument("--dataset", type=str, default="custom_illusion_dataset", help="Dataset (custom_illusion_dataset or kanizsa_square_dataset)")

    args = parser.parse_args()
    
    test_patterns = [p.strip() for p in args.patterns.split(',')]

    config = TestConfig(args.timesteps, args.dataset)
    
    print(f"\n{'='*60}")
    print(f"PATTERN TESTING (SLURM)")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Patterns: {test_patterns}")
    print(f"Device: {config.device}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    all_results = run_pattern_testing(args.models, args.timesteps, test_patterns, config)
    print_pattern_testing_summary(all_results, args.timesteps)
    
    print("\n✓ Pattern testing completed!")

