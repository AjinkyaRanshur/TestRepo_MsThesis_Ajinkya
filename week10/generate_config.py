#!/usr/bin/env python3
"""
Configuration Generator for Batch Experiments
Easily generate experiment configuration files
"""

import json
import argparse
from pathlib import Path

ALL_PATTERNS = [
    "Uniform",
    "Gamma Increasing",
    "Gamma Decreasing",
    "Beta Increasing",
    "Beta Decreasing",
    "Beta Inc & Gamma Dec"
]

def generate_reconstruction_config(patterns, recon_timesteps, iterations, datasetpath, output_file):
    """Generate reconstruction training configuration."""
    config = {
        "experiment_type": "reconstruction",
        "patterns": patterns,
        "recon_timesteps": recon_timesteps,
        "iterations": iterations,
        "datasetpath": datasetpath,
        "log_dir": f"experiment_logs/recon_t{recon_timesteps}"
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created: {output_file}")
    print(f"   Type: Reconstruction")
    print(f"   Patterns: {len(patterns)}")
    print(f"   Timesteps: {recon_timesteps}")

def generate_classification_config(patterns, recon_timesteps, class_timesteps, iterations, output_file):
    """Generate classification training configuration."""
    config = {
        "experiment_type": "classification",
        "patterns": patterns,
        "recon_timesteps": recon_timesteps,
        "class_timesteps": class_timesteps,
        "iterations": iterations,
        "datasetpath": "data/visual_illusion_dataset",
        "log_dir": f"experiment_logs/class_r{recon_timesteps}_c{class_timesteps}"
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created: {output_file}")
    print(f"   Type: Classification")
    print(f"   Patterns: {len(patterns)}")
    print(f"   Timesteps: recon={recon_timesteps}, class={class_timesteps}")

def generate_testing_config(trained_pattern, test_patterns, recon_timesteps, class_timesteps, 
                           test_timesteps, output_file):
    """Generate testing configuration."""
    config = {
        "experiment_type": "testing",
        "trained_pattern": trained_pattern,
        "model_config": {
            "recon": recon_timesteps,
            "class": class_timesteps
        },
        "test_patterns": test_patterns,
        "test_timesteps": test_timesteps,
        "log_dir": f"experiment_logs/test_{trained_pattern.lower().replace(' ', '_')}"
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created: {output_file}")
    print(f"   Type: Testing")
    print(f"   Model: {trained_pattern}")
    print(f"   Test Patterns: {len(test_patterns)}")

def generate_hyperparameter_sweep(base_output_dir, param_name, param_values, exp_type="reconstruction"):
    """Generate multiple configs for hyperparameter sweep."""
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    configs_created = []
    
    for value in param_values:
        if exp_type == "reconstruction":
            output_file = f"{base_output_dir}/recon_{param_name}{value}.json"
            config = {
                "experiment_type": "reconstruction",
                "patterns": ["Uniform"],  # Usually sweep on one pattern
                "recon_timesteps": value if param_name == "t" else 10,
                "iterations": value if param_name == "iter" else 20,
                "datasetpath": "/home/ajinkyar/datasets/",
                "log_dir": f"experiment_logs/sweep_{param_name}{value}"
            }
        elif exp_type == "classification":
            output_file = f"{base_output_dir}/class_{param_name}{value}.json"
            config = {
                "experiment_type": "classification",
                "patterns": ["Uniform"],
                "recon_timesteps": 10,
                "class_timesteps": value if param_name == "t" else 10,
                "iterations": value if param_name == "iter" else 25,
                "datasetpath": "data/visual_illusion_dataset",
                "log_dir": f"experiment_logs/sweep_{param_name}{value}"
            }
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        configs_created.append(output_file)
    
    print(f"\n✅ Created {len(configs_created)} configurations for sweep:")
    print(f"   Parameter: {param_name}")
    print(f"   Values: {param_values}")
    for cfg in configs_created:
        print(f"   - {cfg}")

def main():
    parser = argparse.ArgumentParser(description='Generate experiment configuration files')
    parser.add_argument('--type', choices=['reconstruction', 'classification', 'testing', 'sweep'],
                       required=True, help='Type of experiment')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--patterns', nargs='+', default=ALL_PATTERNS,
                       help='Patterns to use (default: all)')
    parser.add_argument('--all-patterns', action='store_true',
                       help='Use all available patterns')
    
    # Reconstruction options
    parser.add_argument('--recon-timesteps', type=int, default=10,
                       help='Reconstruction timesteps (default: 10)')
    parser.add_argument('--recon-iterations', type=int, default=20,
                       help='Reconstruction iterations (default: 20)')
    parser.add_argument('--datasetpath', default='/home/ajinkyar/datasets/',
                       help='Dataset path for reconstruction')
    
    # Classification options
    parser.add_argument('--class-timesteps', type=int, default=10,
                       help='Classification timesteps (default: 10)')
    parser.add_argument('--class-iterations', type=int, default=25,
                       help='Classification iterations (default: 25)')
    
    # Testing options
    parser.add_argument('--trained-pattern', default='Uniform',
                       help='Pattern the model was trained on (for testing)')
    parser.add_argument('--test-timesteps', type=int, default=10,
                       help='Timesteps for testing (default: 10)')
    parser.add_argument('--test-patterns', nargs='+', default=ALL_PATTERNS,
                       help='Patterns to test on')
    
    # Sweep options
    parser.add_argument('--sweep-param', choices=['t', 'iter'],
                       help='Parameter to sweep (t=timesteps, iter=iterations)')
    parser.add_argument('--sweep-values', type=int, nargs='+',
                       help='Values to sweep over')
    
    args = parser.parse_args()
    
    # Use all patterns if requested
    if args.all_patterns:
        args.patterns = ALL_PATTERNS
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate configuration based on type
    if args.type == 'reconstruction':
        generate_reconstruction_config(
            patterns=args.patterns,
            recon_timesteps=args.recon_timesteps,
            iterations=args.recon_iterations,
            datasetpath=args.datasetpath,
            output_file=args.output
        )
    
    elif args.type == 'classification':
        generate_classification_config(
            patterns=args.patterns,
            recon_timesteps=args.recon_timesteps,
            class_timesteps=args.class_timesteps,
            iterations=args.class_iterations,
            output_file=args.output
        )
    
    elif args.type == 'testing':
        generate_testing_config(
            trained_pattern=args.trained_pattern,
            test_patterns=args.test_patterns,
            recon_timesteps=args.recon_timesteps,
            class_timesteps=args.class_timesteps,
            test_timesteps=args.test_timesteps,
            output_file=args.output
        )
    
    elif args.type == 'sweep':
        if not args.sweep_param or not args.sweep_values:
            print("Error: --sweep-param and --sweep-values required for sweep")
            return 1
        
        generate_hyperparameter_sweep(
            base_output_dir=args.output,
            param_name=args.sweep_param,
            param_values=args.sweep_values,
            exp_type='reconstruction'  # Can be extended
        )
    
    print(f"\n📝 Next steps:")
    print(f"   1. Review the configuration: cat {args.output}")
    print(f"   2. Submit to SLURM: sbatch submit_experiment.sh {args.output}")
    print(f"   3. Monitor progress: ./monitor_progress.sh\n")

if __name__ == '__main__':
    main()
