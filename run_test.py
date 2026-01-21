#!/usr/bin/env python3
"""
Unified test runner for all testing modes
Consolidates run_trajectory_test.py, run_pattern_test.py, run_grid_search_test.py
"""

import argparse
import torch
import numpy as np
from test_workflow import run_trajectory_test
from pattern_testing import run_pattern_testing, print_pattern_testing_summary
from grid_search_testing import run_grid_search, print_grid_search_summary


class TestConfig:
    """Shared test configuration"""
    def __init__(self, timesteps, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 40
        self.timesteps = timesteps
        self.seed = 42

        self.gammaset = [[0.33, 0.33, 0.33, 0.33]]
        self.betaset = [[0.33, 0.33, 0.33, 0.33]]
        self.alphaset = [[0.1, 0.1, 0.1, 0.1]]

        self.classification_datasetpath = dataset
        self.recon_datasetpath = dataset


def run_trajectory(args, config):
    """Run trajectory testing"""
    print(f"\n{'='*60}")
    print(f"TRAJECTORY TESTING")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    run_trajectory_test(args.models, args.timesteps, config)
    print("\n✓ Trajectory testing completed!")


def run_pattern(args, config):
    """Run pattern testing"""
    test_patterns = [p.strip() for p in args.patterns.split(',')]

    print(f"\n{'='*60}")
    print(f"PATTERN TESTING")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Patterns: {test_patterns}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    all_results = run_pattern_testing(args.models, args.timesteps, test_patterns, config)
    print_pattern_testing_summary(all_results, args.timesteps)
    print("\n✓ Pattern testing completed!")


def run_grid(args, config):
    """Run grid search testing"""
    gamma_range = (args.gamma_start, args.gamma_stop, args.gamma_step)
    beta_range = (args.beta_start, args.beta_stop, args.beta_step)

    print(f"\n{'='*60}")
    print(f"GRID SEARCH")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Dataset: {args.dataset}")
    print(f"Gamma range: {gamma_range}")
    print(f"Beta range: {beta_range}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    results = run_grid_search(args.models, args.timesteps, config, gamma_range, beta_range)

    gamma_values = np.arange(*gamma_range)
    beta_values = np.arange(*beta_range)
    print_grid_search_summary(results, gamma_values, beta_values)
    print("\n✓ Grid search completed!")


def main():
    parser = argparse.ArgumentParser(description="Unified test runner for predictive coding models")

    # Common arguments
    parser.add_argument("--mode", type=str, required=True,
                       choices=["trajectory", "pattern", "grid"],
                       help="Testing mode: trajectory, pattern, or grid")
    parser.add_argument("--models", nargs='+', required=True,
                       help="Model names to test")
    parser.add_argument("--timesteps", type=int, required=True,
                       help="Number of test timesteps")
    parser.add_argument("--dataset", type=str, default="custom_illusion_dataset",
                       help="Dataset (custom_illusion_dataset or kanizsa_square_dataset)")

    # Pattern-specific arguments
    parser.add_argument("--patterns", type=str,
                       help="Comma-separated pattern names (for pattern mode)")

    # Grid search-specific arguments
    parser.add_argument("--gamma-start", type=float,
                       help="Gamma range start (for grid mode)")
    parser.add_argument("--gamma-stop", type=float,
                       help="Gamma range stop (for grid mode)")
    parser.add_argument("--gamma-step", type=float,
                       help="Gamma range step (for grid mode)")
    parser.add_argument("--beta-start", type=float,
                       help="Beta range start (for grid mode)")
    parser.add_argument("--beta-stop", type=float,
                       help="Beta range stop (for grid mode)")
    parser.add_argument("--beta-step", type=float,
                       help="Beta range step (for grid mode)")

    args = parser.parse_args()
    config = TestConfig(args.timesteps, args.dataset)

    # Dispatch to appropriate test mode
    if args.mode == "trajectory":
        run_trajectory(args, config)
    elif args.mode == "pattern":
        if not args.patterns:
            parser.error("--patterns is required for pattern mode")
        run_pattern(args, config)
    elif args.mode == "grid":
        required_grid_args = ["gamma_start", "gamma_stop", "gamma_step",
                              "beta_start", "beta_stop", "beta_step"]
        missing = [arg for arg in required_grid_args if getattr(args, arg) is None]
        if missing:
            parser.error(f"Grid mode requires: {', '.join(['--' + arg.replace('_', '-') for arg in missing])}")
        run_grid(args, config)


if __name__ == "__main__":
    main()
