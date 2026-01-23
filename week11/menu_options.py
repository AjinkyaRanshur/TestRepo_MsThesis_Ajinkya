import pyfiglet
from colorama import Fore, Style, init
from utils import clear, banner, parse_list
from model_tracking import get_tracker

# ============================================================
# MAIN MENU FUNCTIONS
# ============================================================

def main_menu():
    """Main menu - choose between SLURM jobs or registry"""
    clear()
    banner("Pred-Net Analyzer")
    print(Fore.YELLOW + "Main Menu:\n")
    print(Fore.GREEN + " [1] SLURM Job Submission (Training & Testing)")
    print(Fore.GREEN + " [2] View Model Registry")
    print(Fore.GREEN + " [3] Interactive Mode (Placeholder)")
    print(Fore.GREEN + " [4] Generate Aggregate Plots (Post-Processing)")
    print(Fore.RED   + " [0] Exit\n")
    return input(Fore.WHITE + "Enter choice: ")


def slurm_job_type_menu():
    """Choose between training or testing job"""
    clear()
    banner("SLURM Job Type")
    print(Fore.YELLOW + "Select job type:\n")
    print(Fore.GREEN + " [1] Training Job")
    print(Fore.GREEN + " [2] Testing Job")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")


# ============================================================
# SLURM TRAINING MENUS
# ============================================================

def slurm_training_type_menu():
    """Select type of training for SLURM"""
    clear()
    banner("Training Type")
    print(Fore.YELLOW + "Select training type:\n")
    print(Fore.GREEN + " [1] Reconstruction Training")
    print(Fore.GREEN + " [2] Classification Training")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")


# ============================================================
# SLURM TESTING MENUS
# ============================================================

def slurm_testing_type_menu():
    """Select what type of test to run for SLURM"""
    clear()
    banner("Testing Type")
    print(Fore.YELLOW + "Select test type:\n")
    print(Fore.GREEN + " [1] Trajectory Testing (PC dynamics over timesteps)")
    print(Fore.GREEN + " [2] Pattern Testing (Test with different patterns)")
    print(Fore.GREEN + " [3] Grid Search Testing (Hyperparameter search)")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")


def slurm_select_test_models():
    """
    Select models for SLURM testing
    Returns list of model names grouped by configuration
    """
    clear()
    banner("Model Selection")
    
    tracker = get_tracker()
    
    # Get completed classification models only
    models = tracker.get_models_by_type("classification_training_shapes")
    models = [m for m in models if m.get('status') == 'completed']
    
    if not models:
        print(Fore.RED + "No completed classification models found!")
        print(Fore.YELLOW + "Train classification models first.")
        input("\nPress ENTER...")
        return None
    
    # Group by configuration (excluding seed)
    config_groups = {}
    for model in models:
        config = model['config']
        
        key = (
            config.get('pattern'),
            config.get('timesteps'),
            config.get('Dataset'),
            config.get('base_recon_model'),
            config.get('checkpoint_epoch')
        )
        
        if key not in config_groups:
            config_groups[key] = []
        config_groups[key].append(model['name'])
    
    # Display configurations
    print(Fore.GREEN + f"Found {len(config_groups)} unique model configurations:\n")
    
    configs_list = list(config_groups.items())
    for i, (config_key, model_names) in enumerate(configs_list, 1):
        pattern, timesteps, dataset, base_model, chk_epoch = config_key
        print(f"{i}. Pattern: {pattern}, Timesteps: {timesteps}, Dataset: {dataset}")
        print(f"   Base: {base_model}, Checkpoint: {chk_epoch}")
        print(f"   Seeds: {len(model_names)} ({', '.join([m.split('_s')[-1] for m in model_names])})\n")
    
    # Select configuration
    print(Fore.YELLOW + "Select configuration number (or 'all' for all):")
    choice = input(Fore.WHITE + "Your choice: ").strip()
    
    if choice.lower() == 'all':
        selected_models = []
        for model_names in config_groups.values():
            selected_models.extend(model_names)
        return selected_models
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(configs_list):
            selected_config = configs_list[idx]
            return selected_config[1]  # Return list of model names
        else:
            print(Fore.RED + "Invalid selection!")
            return None
    except ValueError:
        print(Fore.RED + "Invalid input!")
        return None


def slurm_get_test_timesteps():
    """Get timesteps for SLURM testing"""
    clear()
    banner("Test Configuration")
    print(Fore.YELLOW + "Enter test timesteps:\n")
    timesteps_input = input("Timesteps (default 10): ").strip() or "10"
    return int(timesteps_input)


def slurm_get_test_patterns():
    """Get patterns for SLURM pattern testing"""
    clear()
    banner("Pattern Selection")
    
    patterns = [
        "Uniform", 
        "Gamma Increasing", 
        "Gamma Decreasing", 
        "Beta Increasing", 
        "Beta Decreasing",
        "Beta Inc & Gamma Dec"
    ]
    
    print(Fore.YELLOW + "Available patterns:\n")
    for i, p in enumerate(patterns, 1):
        print(f"{Fore.GREEN}  {i}. {p}")
    
    print(Fore.YELLOW + "\nSelect patterns to test:")
    print("  - Enter pattern numbers (comma-separated, e.g., 1,2,3)")
    print("  - Or enter 'all' for all patterns\n")
    
    choice = input(Fore.WHITE + "Your choice: ").strip()
    
    if choice.lower() == 'all':
        return patterns
    
    try:
        indices = [int(x.strip())-1 for x in choice.split(',')]
        selected = [patterns[i] for i in indices if 0 <= i < len(patterns)]
        return selected if selected else None
    except (ValueError, IndexError):
        print(Fore.RED + "Invalid selection!")
        return None


def slurm_get_grid_search_params():
    """Get grid search parameters for SLURM"""
    clear()
    banner("Grid Search Setup")
    
    print(Fore.YELLOW + "Configure grid search parameters:\n")
    
    # Gamma range
    print("Gamma range:")
    gamma_start = float(input("  Start (default 0.13): ").strip() or "0.13")
    gamma_stop = float(input("  Stop (default 0.53): ").strip() or "0.53")
    gamma_step = float(input("  Step (default 0.1): ").strip() or "0.1")
    
    # Beta range
    print("\nBeta range:")
    beta_start = float(input("  Start (default 0.13): ").strip() or "0.13")
    beta_stop = float(input("  Stop (default 0.53): ").strip() or "0.53")
    beta_step = float(input("  Step (default 0.1): ").strip() or "0.1")
    
    return {
        "gamma_range": (gamma_start, gamma_stop, gamma_step),
        "beta_range": (beta_start, beta_stop, beta_step)
    }


# ============================================================
# SLURM BATCH SUBMISSION
# ============================================================

def slurm_recon_entries():
    """
    Collect parameters for reconstruction training batch submission
    """
    clear()
    banner("Reconstruction Training")
    
    print(Fore.YELLOW + "SLURM BATCH SUBMISSION - RECONSTRUCTION\n")
    print("Enter comma-separated values for multiple parameters:\n")
    
    # Get epochs
    epochs_input = input("Epochs (default 200): ").strip() or "200"
    epochs = parse_list(epochs_input, int)

    # Get dataset - ✅ FIXED: Now handles ranges and multiple selections
    print("\nAvailable datasets:")
    print("  1. cifar10")
    print("  2. stl10")
    print("  3. custom_illusion_dataset")
    print("  4. kanizsa_square_dataset")
    dataset_input = input("Dataset choice (comma-separated or range like 1-3, default cifar10): ").strip() or "1"
    
    dataset_map = {
        "1": "cifar10", 
        "2": "stl10", 
        "3": "custom_illusion_dataset",
        "4": "kanizsa_square_dataset"
    }
    
    # ✅ Parse dataset selection (handles comma-separated and ranges)
    dataset_list = []
    if dataset_input:
        for part in dataset_input.split(','):
            part = part.strip()
            if '-' in part:
                # Handle range like "1-3"
                try:
                    start, end = part.split('-')
                    for i in range(int(start), int(end) + 1):
                        if str(i) in dataset_map and dataset_map[str(i)] not in dataset_list:
                            dataset_list.append(dataset_map[str(i)])
                except ValueError:
                    print(f"{Fore.YELLOW}Warning: Invalid range '{part}', skipping{Fore.RESET}")
            else:
                # Handle single selection
                if part in dataset_map and dataset_map[part] not in dataset_list:
                    dataset_list.append(dataset_map[part])
    
    # Default to cifar10 if nothing valid selected
    if not dataset_list:
        dataset_list = ["cifar10"]
    
    print(f"{Fore.GREEN}Selected datasets: {', '.join(dataset_list)}{Fore.RESET}\n")
    
    # Get batch size
    batch_input = input("Batch size (default 40): ").strip() or "40"
    batch_size = parse_list(batch_input, int)
    
    # Get learning rate
    lr_input = input("Learning rate (default 0.00005): ").strip() or "0.00005"
    lr = parse_list(lr_input, float)
    
    # Get timesteps
    timesteps_input = input("Timesteps (default 10): ").strip() or "10"
    timesteps = parse_list(timesteps_input, int)
    
    # Select patterns
    print("\nAvailable patterns:")
    patterns = [
        "Uniform", 
        "Gamma Increasing", 
        "Gamma Decreasing", 
        "Beta Increasing", 
        "Beta Decreasing",
        "Beta Inc & Gamma Dec"
    ]
    
    for i, p in enumerate(patterns, 1):
        print(f"  {i}. {p}")
    
    pattern_choice = input("\nSelect patterns (comma-separated numbers, or 'all'): ").strip()
    
    if pattern_choice.lower() == 'all':
        selected_patterns = patterns
    else:
        indices = [int(x.strip())-1 for x in pattern_choice.split(',')]
        selected_patterns = [patterns[i] for i in indices if 0 <= i < len(patterns)]
    
    # Get seeds
    num_seeds_input = input("\nHow many random seeds? (default 3): ").strip() or "3"
    num_seeds = int(num_seeds_input)
    
    import random
    seeds = [random.randint(1, 10000) for _ in range(num_seeds)]
    
    print(f"\nGenerated seeds: {seeds}")
    
    # Calculate total models
    number_of_models = (
        len(epochs) * 
        len(batch_size) * 
        len(lr) * 
        num_seeds * 
        len(timesteps) *  
        len(selected_patterns) *
        len(dataset_list)
    )

    print(f"\n{'='*60}")
    print(f"Will create {number_of_models} reconstruction models")
    print(f"{'='*60}")
    print(f"Patterns: {len(selected_patterns)}")
    print(f"Seeds: {num_seeds}")
    print(f"Learning rates: {len(lr)}")
    print(f"Timesteps: {len(timesteps)}")
    print(f"Epochs: {len(epochs)}")
    print(f"Datasets: {len(dataset_list)}")
    print(f"{'='*60}\n")
    
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        return None
    
    return {
        "train_cond": "recon_pc_train",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "timesteps": timesteps,
        "selected_patterns": selected_patterns,
        "seeds": seeds,
        "dataset_list": dataset_list,
    }


def slurm_classification_entries():
    """
    Collect parameters for classification training batch submission
    """
    clear()
    banner("Classification Training")
    
    print(Fore.YELLOW + "SLURM BATCH SUBMISSION - CLASSIFICATION\n")
    
    # Show available reconstruction models
    tracker = get_tracker()
    completed_models = tracker.get_completed_recon_models()
    
    if not completed_models:
        print(Fore.RED + "ERROR: No completed reconstruction models!")
        print(Fore.YELLOW + "Train reconstruction models first.")
        input("\nPress ENTER...")
        return None
    
    print(Fore.GREEN + f"Found {len(completed_models)} completed reconstruction models:\n")
    for i, model in enumerate(completed_models, 1):
        config = model.get('config', {})
        print(f"  {i}. {model['name']}")
        print(f"     Pattern: {config.get('pattern')}, Seed: {config.get('seed')}, "
              f"Timesteps: {config.get('timesteps')}, Dataset: {config.get('Dataset')}\n")
    
    # Select base models
    print(Fore.YELLOW + "Select base reconstruction models:")
    print("  - Enter model numbers (comma-separated, e.g., 1,2,3)")
    print("  - Or enter 'all' to use all models\n")
    
    model_choice = input(Fore.WHITE + "Your choice: ").strip()
    
    if model_choice.lower() == 'all':
        selected_models = [m['name'] for m in completed_models]
    else:
        try:
            indices = [int(x.strip())-1 for x in model_choice.split(',')]
            selected_models = [completed_models[i]['name'] for i in indices 
                             if 0 <= i < len(completed_models)]
        except (ValueError, IndexError):
            print(Fore.RED + "Invalid selection!")
            input("Press ENTER...")
            return None
    
    print(f"\n{Fore.GREEN}Selected {len(selected_models)} base models{Fore.RESET}")
   
    # NEW: Ask which dataset to use for classification training
    print(Fore.YELLOW + "\nWhich dataset to use for classification training?")
    print("  1. custom_illusion_dataset (6 classes: 5 shapes + random)")
    print("  2. kanizsa_square_dataset (2 classes: square vs random)")

    dataset_choice = input("Dataset choice (1-2, default 1): ").strip() or "1"

    if dataset_choice == "2":
       classification_dataset = "kanizsa_square_dataset"
       number_of_classes = [2]  # square vs random
    else:
       classification_dataset = "custom_illusion_dataset"
       number_of_classes = [6]  # 5 shapes + random

    dataset_list = [classification_dataset]

    print(f"\n{Fore.GREEN}Classification dataset: {classification_dataset}{Fore.RESET}")
    print(f"{Fore.GREEN}Number of classes: {number_of_classes[0]}{Fore.RESET}")    


    # Select checkpoint epochs
    print(Fore.YELLOW + "\nWhich checkpoint epochs to use?")
    print("  Checkpoints saved every 10 epochs (10, 20, 30, ...)")
    checkpoint_input = input("Enter epochs (comma-separated, e.g., 50,100,150): ").strip()
    checkpoint_epochs = parse_list(checkpoint_input, int)
    
    # Get training parameters
    print("\nTraining Parameters:")
    
    epochs_input = input("Epochs (default 25): ").strip() or "25"
    epochs = parse_list(epochs_input, int)
    
    batch_input = input("Batch size (default 40): ").strip() or "40"
    batch_size = parse_list(batch_input, int)
    
    lr_input = input("Learning rate (default 0.00005): ").strip() or "0.00005"
    lr = parse_list(lr_input, float)
    
    timesteps_input = input("Timesteps (default 10): ").strip() or "10"
    timesteps = parse_list(timesteps_input, int)
    
    # Select patterns
    print("\nClassification patterns:")
    patterns = [
        "Uniform", 
        "Gamma Increasing", 
        "Gamma Decreasing", 
        "Beta Increasing", 
        "Beta Decreasing",
        "Beta Inc & Gamma Dec"
    ]
    
    for i, p in enumerate(patterns, 1):
        print(f"  {i}. {p}")
    
    pattern_choice = input("\nSelect patterns (comma-separated, or 'all'): ").strip()
    
    if pattern_choice.lower() == 'all':
        selected_patterns = patterns
    else:
        indices = [int(x.strip())-1 for x in pattern_choice.split(',')]
        selected_patterns = [patterns[i] for i in indices if 0 <= i < len(patterns)]
    
    # Optimizer scope
    print(Fore.YELLOW + "\nOptimizer scope:")
    print("  [1] Linear layers only (fc1, fc2, fc3)")
    print("  [2] All layers (conv + linear)")
    opt_choice = input("Choice (default 1): ").strip() or "1"
    optimize_all_layers = (opt_choice == "2")
    
    # Calculate total models
    number_of_models = (
        len(selected_models) *
        len(checkpoint_epochs) *
        len(epochs) * 
        len(batch_size) * 
        len(lr) * 
        len(timesteps) * 
        len(selected_patterns)
    )

    print(f"\n{'='*60}")
    print(f"Will create {number_of_models} classification models")
    print(f"{'='*60}")
    print(f"Base Models: {len(selected_models)}")
    print(f"Checkpoints per model: {len(checkpoint_epochs)}")
    print(f"Patterns: {len(selected_patterns)}")
    print(f"Timesteps: {len(timesteps)}")
    print(f"Optimize all layers: {optimize_all_layers}")
    print(f"Classification Dataset: {classification_dataset}")
    print(f"Number of classes: {number_of_classes[0]}")
    print(f"{'='*60}\n")
    
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        return None
    
    return {
        "train_cond": "classification_training_shapes",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "timesteps": timesteps,
        "number_of_classes": number_of_classes,
        "selected_patterns": selected_patterns,
        "base_recon_models": selected_models,
        "checkpoint_epochs": checkpoint_epochs,
        "dataset_list": dataset_list,
        "optimize_all_layers": optimize_all_layers
    }


# ============================================================
# REGISTRY VIEWER
# ============================================================

def view_registry():
    """View all models in registry"""
    clear()
    banner("Model Registry")
    
    tracker = get_tracker()
    
    print(Fore.YELLOW + "Filter by:\n")
    print(Fore.GREEN + " [1] All models")
    print(Fore.GREEN + " [2] Reconstruction models")
    print(Fore.GREEN + " [3] Classification models")
    print(Fore.GREEN + " [4] Completed models only")
    print(Fore.RED   + " [0] Back\n")
    
    choice = input(Fore.WHITE + "Enter choice: ").strip()
    
    if choice == "0":
        return
    
    if choice == "1":
        models = tracker.list_all_models()
        title = "ALL MODELS"
    elif choice == "2":
        models = tracker.get_models_by_type("recon_pc_train")
        title = "RECONSTRUCTION MODELS"
    elif choice == "3":
        models = tracker.get_models_by_type("classification_training_shapes")
        title = "CLASSIFICATION MODELS"
    elif choice == "4":
        models = tracker.list_all_models(filter_status="completed")
        title = "COMPLETED MODELS"
    else:
        print(Fore.RED + "Invalid choice!")
        input("Press ENTER...")
        return
    
    clear()
    print(Fore.CYAN + f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    if not models:
        print(Fore.YELLOW + "No models found.")
    else:
        for i, model in enumerate(models, 1):
            config = model.get('config', {})
            print(Fore.GREEN + f"{i}. {model['name']}")
            print(Fore.WHITE + f"   Status: {model.get('status')}")
            print(Fore.WHITE + f"   Pattern: {config.get('pattern')}, Seed: {config.get('seed')}")
            print(Fore.WHITE + f"   Created: {model.get('created_at')}")
            
            if model.get('checkpoint_path'):
                print(Fore.WHITE + f"   Checkpoint: {model['checkpoint_path']}")
            
            print()
    
    input(Fore.YELLOW + "\nPress ENTER to continue...")


# ============================================================
# POST-PROCESSING AGGREGATION
# ============================================================

def run_post_processing():
    """Run post-processing aggregation for completed models"""
    clear()
    banner("Post-Processing")
    
    print(Fore.YELLOW + "Generate aggregate plots for seed groups:\n")
    print(Fore.GREEN + " [1] Aggregate all completed model groups")
    print(Fore.GREEN + " [2] Aggregate specific models")
    print(Fore.RED   + " [0] Back\n")
    
    choice = input(Fore.WHITE + "Enter choice: ").strip()
    
    if choice == "0":
        return
    
    if choice == "1":
        # Aggregate all
        print(f"\n{Fore.CYAN}Running post-processing aggregation...")
        import os
        os.system("python post_training_aggregation.py")
        input("\nPress ENTER to continue...")
    
    elif choice == "2":
        # Select specific models
        tracker = get_tracker()
        completed = tracker.list_all_models(filter_status="completed")
        
        if not completed:
            print(Fore.RED + "No completed models found!")
            input("Press ENTER...")
            return
        
        print(f"\n{Fore.GREEN}Completed models:\n")
        for i, model in enumerate(completed, 1):
            print(f"  {i}. {model['name']}")
        
        print(Fore.YELLOW + "\nEnter model numbers (comma-separated):")
        model_choice = input(Fore.WHITE + "Your choice: ").strip()
        
        try:
            indices = [int(x.strip())-1 for x in model_choice.split(',')]
            selected = [completed[i]['name'] for i in indices if 0 <= i < len(completed)]
            
            if selected:
                model_str = " ".join(selected)
                print(f"\n{Fore.CYAN}Running aggregation for {len(selected)} models...")
                import os
                os.system(f"python post_training_aggregation.py --models {model_str}")
            else:
                print(Fore.RED + "No valid models selected!")
        except (ValueError, IndexError):
            print(Fore.RED + "Invalid selection!")
        
        input("\nPress ENTER to continue...")
    
    else:
        print(Fore.RED + "Invalid choice!")
        input("Press ENTER...")


# ============================================================
# INTERACTIVE MODE (PLACEHOLDER)
# ============================================================

def interactive_mode_placeholder():
    """Placeholder for interactive mode"""
    clear()
    banner("Interactive Mode")
    
    print(Fore.YELLOW + "Interactive mode is currently under development.\n")
    print(Fore.CYAN + "This mode will allow you to:")
    print("  • Train single models interactively")
    print("  • Test models in real-time")
    print("  • View results immediately")
    print("\nFor now, please use SLURM Job Submission for training and testing.\n")
    
    input(Fore.WHITE + "Press ENTER to return to main menu...")
