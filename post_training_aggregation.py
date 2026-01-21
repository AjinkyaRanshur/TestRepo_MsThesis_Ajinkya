"""
Post-Training Aggregation Script
Run this AFTER all SLURM training jobs complete to generate aggregate plots

Usage:
    # Aggregate specific models
    python post_training_aggregation.py --models model1 model2 model3
    
    # Aggregate all completed model groups
    python post_training_aggregation.py
"""

import argparse
from model_tracking import get_tracker
from eval_and_plotting import plot_training_metrics_with_seeds
from colorama import Fore, init

init(autoreset=True)


def aggregate_specific_models(model_names):
    """
    Generate aggregate plot for specific list of models
    """
    tracker = get_tracker()
    
    # Verify all models exist and are completed
    valid_models = []
    for model_name in model_names:
        model_info = tracker.get_model(model_name)
        if not model_info:
            print(f"{Fore.RED}⚠ Model not found: {model_name}")
            continue
        
        if model_info.get('status') != 'completed':
            print(f"{Fore.YELLOW}⚠ Model not completed: {model_name} (status: {model_info.get('status')})")
            continue
        
        if not model_info.get('metrics') or 'train_loss' not in model_info['metrics']:
            print(f"{Fore.YELLOW}⚠ No metrics found for: {model_name}")
            continue
        
        valid_models.append(model_name)
    
    if not valid_models:
        print(f"{Fore.RED}✗ No valid completed models found")
        return
    
    if len(valid_models) < len(model_names):
        print(f"{Fore.YELLOW}⚠ Only {len(valid_models)}/{len(model_names)} models are valid")
    
    print(f"\n{Fore.GREEN}Generating aggregate plot for {len(valid_models)} models...")
    plot_training_metrics_with_seeds(valid_models)
    print(f"{Fore.GREEN}✓ Aggregate plot generated successfully")


def aggregate_all_completed_models():
    """
    Find all completed model groups and plot aggregates
    Groups models by their configuration (excluding seed)
    """
    tracker = get_tracker()
    
    # Get all completed models
    all_models = tracker.list_all_models(filter_status="completed")
    
    if not all_models:
        print(f"{Fore.YELLOW}⚠ No completed models found in registry")
        return
    
    print(f"\n{Fore.CYAN}Found {len(all_models)} completed models")
    print(f"Grouping by configuration (excluding seed)...")
    
    # Group by configuration (excluding seed)
    config_groups = {}
    
    for model in all_models:
        config = model['config']
        train_cond = config.get('train_cond')
        
        # Skip if no metrics
        if not model.get('metrics') or 'train_loss' not in model['metrics']:
            continue
        
        # Create grouping key based on training condition
        if train_cond == "recon_pc_train":
            key = (
                train_cond,
                config.get('pattern'),
                config.get('timesteps'),
                config.get('Dataset'),
                config.get('lr')
            )
            key_display = f"Recon | {config.get('pattern')} | t={config.get('timesteps')} | {config.get('Dataset')} | lr={config.get('lr')}"
        
        elif train_cond == "classification_training_shapes":
            # UPDATED: Include optimize_all_layers in grouping
            optimize_all = config.get('optimize_all_layers', False)
            key = (
                train_cond,
                config.get('pattern'),
                config.get('timesteps'),
                config.get('Dataset'),
                config.get('lr'),
                config.get('base_recon_model'),
                config.get('checkpoint_epoch'),
                optimize_all
            )
            opt_display = "All" if optimize_all else "Linear"
            key_display = f"Class | {config.get('pattern')} | t={config.get('timesteps')} | Base={config.get('base_recon_model')} | chk={config.get('checkpoint_epoch')} | Opt={opt_display}"
        
        else:
            continue
        
        if key not in config_groups:
            config_groups[key] = {
                'models': [],
                'display': key_display
            }
        config_groups[key]['models'].append(model['name'])
    
    # Plot aggregates for groups with multiple seeds
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"GENERATING AGGREGATE PLOTS")
    print(f"{'='*60}\n")
    
    groups_with_multiple_seeds = 0
    single_seed_groups = 0
    
    for config_key, group_data in config_groups.items():
        model_names = group_data['models']
        
        if len(model_names) > 1:
            groups_with_multiple_seeds += 1
            
            print(f"{Fore.GREEN}Group {groups_with_multiple_seeds}:")
            print(f"  Config: {group_data['display']}")
            print(f"  Models: {len(model_names)} seeds")
            for model_name in model_names:
                print(f"    • {model_name}")
            print(f"  {Fore.YELLOW}Generating aggregate plot...")
            
            try:
                plot_training_metrics_with_seeds(model_names)
                print(f"  {Fore.GREEN}✓ Aggregate plot generated\n")
            except Exception as e:
                print(f"  {Fore.RED}✗ Error generating plot: {e}\n")
        else:
            single_seed_groups += 1
    
    print(f"{Fore.CYAN}{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total model groups: {len(config_groups)}")
    print(f"  Groups with multiple seeds: {groups_with_multiple_seeds} (plots generated)")
    print(f"  Groups with single seed: {single_seed_groups} (no aggregate needed)")
    print(f"{'='*60}\n")
    
    if groups_with_multiple_seeds == 0:
        print(f"{Fore.YELLOW}⚠ No model groups with multiple seeds found")
        print(f"  Train models with different seeds to generate aggregate plots")
    else:
        print(f"{Fore.GREEN}✓ Aggregate plotting complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregate plots for models trained with multiple seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Aggregate all completed model groups automatically
    python post_training_aggregation.py
    
    # Aggregate specific models
    python post_training_aggregation.py --models model1 model2 model3
        """
    )
    parser.add_argument(
        "--models", 
        nargs='+', 
        help="Specific models to aggregate (space-separated)"
    )
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}{'='*60}")
    print(f"POST-TRAINING AGGREGATION")
    print(f"{'='*60}\n")
    
    if args.models:
        # Aggregate specific models
        print(f"Mode: Aggregate specific models")
        print(f"Models: {len(args.models)}")
        for model in args.models:
            print(f"  • {model}")
        print()
        
        aggregate_specific_models(args.models)
    else:
        # Aggregate all completed model groups
        print(f"Mode: Aggregate all completed model groups\n")
        aggregate_all_completed_models()


if __name__ == "__main__":
    main()
