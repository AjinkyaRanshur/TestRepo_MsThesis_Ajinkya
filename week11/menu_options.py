import pyfiglet
from colorama import Fore, Style, init
from utils import clear,banner,parse_list

# ============================================================
# MENU FUNCTIONS
# ============================================================
def main_menu():
    clear()
    banner("Pred-Net Analyzer")
    print(Fore.YELLOW + "Select an option:\n")
    print(Fore.GREEN + " [1] Train the Model")
    print(Fore.GREEN + " [2] Test the Model")
    print(Fore.RED   + " [0] Exit\n")
    return input(Fore.WHITE + "Enter choice: ")

def train_menu():
    clear()
    banner("Training")
    print(Fore.YELLOW + "Training Options:\n")
    print(Fore.GREEN + " [1] Reconstruction Training")
    print(Fore.GREEN + " [2] Classification Training")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def classification_train_menu():
    clear()
    banner("Classification Training")
    print(Fore.YELLOW + "Training Options:\n")
    print(Fore.GREEN + " [1] Training On Specific Pattern")
    print(Fore.GREEN + " [2] Training With All Patterns")
    print(Fore.GREEN + " [3] Training With Learnable Hyperparameters")
    print(Fore.GREEN + " [4] Training with Weighting Over Timesteps")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def test_menu():
    clear()
    banner("Testing")
    print(Fore.YELLOW + "Testing Options:\n")
    print(Fore.GREEN + " [1] Reconstruction Models")
    print(Fore.GREEN + " [2] Classification Models")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def classification_test_menu():
    clear()
    banner("Classification Testing")
    print(Fore.YELLOW + "Testing Options:\n")
    print(Fore.GREEN + " [1] Test Single Model (One Pattern)")
    print(Fore.GREEN + " [2] Test Single Model (All Patterns)")
    print(Fore.GREEN + " [3] Test All Models (All Patterns)")
    print(Fore.GREEN + " [4] Grid Search on Hyperparameters")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def job_running():
    clear()
    banner("ILLUSORY PRED-NET !!")
    print(Fore.YELLOW + "How do you want to run these experiments ?:\n")
    print(Fore.GREEN + " [1] Slurm Submmission (Multiple Experiments)")
    print(Fore.GREEN + " [2] Interactive Running")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

# Add this to your existing slurm_entries function or modify job_running menu
def slurm_training_type():
    """Enhanced job running menu with classification support"""
    clear()
    banner("Experiment Setup")
    print(Fore.YELLOW + "What type of training?\n")
    print(Fore.GREEN + " [1] Reconstruction Training (CIFAR-10)")
    print(Fore.GREEN + " [2] Classification Training (Illusion Dataset)")
    print(Fore.GREEN + " [3] Testing of Trained Models")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def slurm_entries():
    """
    Collect parameters for batch SLURM submission
    Returns a dictionary with all configuration parameters
    """
    clear()
    banner("ILLUSORY PRED-NET !!")
    
    print(Fore.YELLOW + "BATCH SUBMISSION\n")
    
    print("\nTraining Conditions")
    training_condition = input("Training condition (recon_pc_train/classification_training_shapes): ").strip()
   
    print("\nBase Configurations - If Multiple Parameters Need to be tested, use comma-separated values:")
    
    # Get epochs
    epochs_input = input("Epochs (default 200): ").strip() or "200"
    epochs = parse_list(epochs_input, int)
    
    # Get batch size
    batch_input = input("Batch size (default 40): ").strip() or "40"
    batch_size = parse_list(batch_input, int)
    
    # Get learning rate
    lr_input = input("Learning rate (default 0.00005): ").strip() or "0.00005"
    lr = parse_list(lr_input, float)
    
    # Get timesteps
    timesteps_input = input("Timesteps (default 10): ").strip() or "10"
    timesteps = parse_list(timesteps_input, int)
    
    # Get number of classes
    number_of_classes = int(input("Enter Number of classes (10 for CIFAR, 6 for Illusory dataset): ").strip() or "6")
    
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
        print(f"{i}. {p}")
    
    pattern_choice = input("\nSelect patterns (comma-separated numbers, or 'all'): ").strip()
    
    if pattern_choice.lower() == 'all':
        selected_patterns = patterns
    else:
        indices = [int(x.strip())-1 for x in pattern_choice.split(',')]
        selected_patterns = [patterns[i] for i in indices if 0 <= i < len(patterns)]
    
    # Select seeds
    seed_input = input("\nEnter seeds (comma-separated, e.g., 42,123,456): ").strip()
    seeds = [int(x.strip()) for x in seed_input.split(',')]
    
    # Calculate total number of models
    number_of_models = (
        len(epochs) * 
        len(batch_size) * 
        len(lr) * 
        len(timesteps) * 
        len(seeds) * 
        len(selected_patterns)
    )

    print(f"\n{'='*60}")
    print(f"Will create {number_of_models} models")
    print(f"{'='*60}")
    print(f"Patterns: {len(selected_patterns)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Learning rates: {len(lr)}")
    print(f"Timesteps: {len(timesteps)}")
    print(f"Epochs: {len(epochs)}")
    print(f"Batch sizes: {len(batch_size)}")
    print(f"{'='*60}\n")
    
    confirm = input("Proceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Configuration cancelled.")
        return None
    
    base_config = {
        "train_cond": training_condition,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "timesteps": timesteps,
        "number_of_classes": number_of_classes,
        "selected_patterns": selected_patterns,
        "seeds": seeds
    }
    
    return base_config


def slurm_classification_entries():
    """
    Collect parameters for classification training batch submission
    Returns a dictionary with all configuration parameters including base models
    """
    clear()
    banner("ILLUSORY PRED-NET")

    print(Fore.YELLOW + "CLASSIFICATION TRAINING\n")

    
    print("\nClassification Training Setup")
    print("="*60)
    
    # First, show available reconstruction models
    from model_tracking import get_tracker
    tracker = get_tracker()
    completed_models = tracker.get_completed_recon_models()
    
    if not completed_models:
        print(Fore.RED + "ERROR: No completed reconstruction models found!")
        print(Fore.YELLOW + "You must train reconstruction models first.")
        input("\nPress ENTER to continue...")
        return None
    
    print(Fore.GREEN + f"\nFound {len(completed_models)} completed reconstruction models:")
    for i, model in enumerate(completed_models, 1):
        config = model.get('config', {})
        print(f"  {i}. {model['name']} (Pattern: {config.get('pattern')}, Seed: {config.get('seed')})")
    
    # Select base models
    print(Fore.YELLOW + "\nSelect base reconstruction models:")
    print("  - Enter model numbers (comma-separated, e.g., 1,2,3)")
    print("  - Or enter 'all' to use all models")
    
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
    
    # Select checkpoint epochs
    print(Fore.YELLOW + "\nWhich checkpoint epochs to use?")
    print("  Checkpoints are saved every 10 epochs (1, 2, 3, ... = epochs 10, 20, 30, ...)")
    checkpoint_input = input("Enter checkpoint indices (comma-separated, e.g., 10,15,20): ").strip()
    checkpoint_epochs = parse_list(checkpoint_input, int)
    
    print("\nBase Configurations:")
    
    # Get epochs
    epochs_input = input("Epochs (default 25): ").strip() or "25"
    epochs = parse_list(epochs_input, int)
    
    # Get batch size
    batch_input = input("Batch size (default 40): ").strip() or "40"
    batch_size = parse_list(batch_input, int)
    
    # Get learning rate
    lr_input = input("Learning rate (default 0.00005): ").strip() or "0.00005"
    lr = parse_list(lr_input, float)
    
    # Get timesteps
    timesteps_input = input("Timesteps (default 10): ").strip() or "10"
    timesteps = parse_list(timesteps_input, int)
    
    # Number of classes
    number_of_classes = 6  # Fixed for illusion dataset
    
    # Select patterns for testing
    print("\nTrain on pattern (for classification models):")
    patterns = [
        "Uniform", 
        "Gamma Increasing", 
        "Gamma Decreasing", 
        "Beta Increasing", 
        "Beta Decreasing",
        "Beta Inc & Gamma Dec"
    ]
    
    for i, p in enumerate(patterns, 1):
        print(f"{i}. {p}")
    
    pattern_choice = input("\nSelect patterns (comma-separated numbers, or 'all'): ").strip()
    
    if pattern_choice.lower() == 'all':
        selected_patterns = patterns
    else:
        indices = [int(x.strip())-1 for x in pattern_choice.split(',')]
        selected_patterns = [patterns[i] for i in indices if 0 <= i < len(patterns)]
    
    # Select seeds
    seed_input = input("\nEnter seeds (comma-separated, e.g., 42,123,456): ").strip()
    seeds = [int(x.strip()) for x in seed_input.split(',')]
    
    # Calculate total number of models
    number_of_models = (
        len(selected_models) *
        len(checkpoint_epochs) *
        len(epochs) * 
        len(batch_size) * 
        len(lr) * 
        len(timesteps) * 
        len(seeds) * 
        len(selected_patterns)
    )

    print(f"\n{'='*60}")
    print(f"Will create {number_of_models} classification models")
    print(f"{'='*60}")
    print(f"Base Models: {len(selected_models)}")
    print(f"Checkpoints per model: {len(checkpoint_epochs)}")
    print(f"Patterns: {len(selected_patterns)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Learning rates: {len(lr)}")
    print(f"Timesteps: {len(timesteps)}")
    print(f"Epochs: {len(epochs)}")
    print(f"Batch sizes: {len(batch_size)}")
    print(f"{'='*60}\n")
    
    confirm = input("Proceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Configuration cancelled.")
        return None
    
    base_config = {
        "train_cond": "classification_training_shapes",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "timesteps": timesteps,
        "number_of_classes": number_of_classes,
        "selected_patterns": selected_patterns,
        "seeds": seeds,
        "base_recon_models": selected_models,
        "checkpoint_epochs": checkpoint_epochs
    }
    
    return base_config

def slurm_testing_entries():
    """
    Collect parameters for classification testing batch submission
    Returns a dictionary with all configuration parameters including base models
    """
    clear()
    banner("ILLUSORY PRED-NET !!")

    print(Fore.YELLOW + "TESTING OF MODELS\n")
    
    print("\nClassification Training Setup")
    print("="*60)
    
    # First, show available reconstruction models
    from model_tracking import get_tracker
    tracker = get_tracker()
    completed_models = tracker.get_completed_classification_models()
    
    if not completed_models:
        print(Fore.RED + "ERROR: No completed Classification models found!")
        print(Fore.YELLOW + "You must train classification models first.")
        input("\nPress ENTER to continue...")
        return None
    
    print(Fore.GREEN + f"\nFound {len(completed_models)} completed classification models:")
    for i, model in enumerate(completed_models, 1):
        config = model.get('config', {})
        print(f"  {i}. {model['name']} (Pattern: {config.get('pattern')}, Seed: {config.get('seed')})")
    
    # Select base models
    print(Fore.YELLOW + "\nSelect base classification models:")
    print("  - Enter model numbers (comma-separated, e.g., 1,2,3)")
    print("  - Or enter 'all' to use all models")
    
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
    
    # Select checkpoint epochs
    print(Fore.YELLOW + "\nWhich checkpoint epochs to use?")
    print("  Checkpoints are saved every 10 epochs (1, 2, 3, ... = epochs 10, 20, 30, ...)")
    checkpoint_input = input("Enter checkpoint indices (comma-separated, e.g., 10,15,20): ").strip()
    checkpoint_epochs = parse_list(checkpoint_input, int)
    
    print("\nBase Configurations:")
    
    # Get epochs
    #epochs_input = input("Epochs (default 25): ").strip() or "25"
    #epochs = parse_list(epochs_input, int)
   
    
    # Get timesteps
    timesteps_input = input("Timesteps (default 10): ").strip() or "10"
    timesteps = parse_list(timesteps_input, int)

    # Get epochs
    #epochs_input = input("Epochs (default 25): ").strip() or "25"
    #epochs = parse_list(epochs_input, int)

    # Get epochs
    pattern_testing = input("Do you want to perform Pattern Testing on these Models Yes or No (Default No)").strip() or None
   
    # Get epochs
    grid_testing = input("Do you want to perform Grid Search on these Models,Options: single_Layer,all_layers,no (Default No)").strip() or None
  
  
    # Number of classes
    number_of_classes = 6  # Fixed for illusion dataset
    
    # Select patterns for testing
    print("\nTesting patterns (patterns to test the trained classification models):")
    patterns = [
        "Uniform", 
        "Gamma Increasing", 
        "Gamma Decreasing", 
        "Beta Increasing", 
        "Beta Decreasing",
        "Beta Inc & Gamma Dec"
    ]
    
    for i, p in enumerate(patterns, 1):
        print(f"{i}. {p}")
    
    pattern_choice = input("\nSelect patterns (comma-separated numbers, or 'all'): ").strip()
    
    if pattern_choice.lower() == 'all':
        selected_patterns = patterns
    else:
        indices = [int(x.strip())-1 for x in pattern_choice.split(',')]
        selected_patterns = [patterns[i] for i in indices if 0 <= i < len(patterns)]
    
    
    # Calculate total number of models
    number_of_models = (
        len(selected_models) *
        len(checkpoint_epochs) * 
        len(timesteps) * 
        len(selected_patterns)
    )

    print(f"\n{'='*60}")
    print(f"Will Test {number_of_models} classification models")
    print(f"{'='*60}")
    print(f"Base Models: {len(selected_models)}")
    print(f"Checkpoints per model: {len(checkpoint_epochs)}")
    print(f"Trajectory Patterns: {len(selected_patterns)}")
    print(f"Pattern Testing: {len(seeds)}")
    print(f"Grid Search Testing: {len(seeds)}")
    print(f"Timesteps: {len(timesteps)}")
    print(f"{'='*60}\n")
    
    confirm = input("Proceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Configuration cancelled.")
        return None
    
    base_config = {
        "train_cond": "illusion_testing",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "timesteps": timesteps,
        "number_of_classes": number_of_classes,
        "selected_patterns": selected_patterns,
        "seeds": seeds,
        "base_recon_models": selected_models,
        "checkpoint_epochs": checkpoint_epochs
    }
    
    return base_config



def model_selection_menu():
    """Display available reconstruction models for classification training"""
    from model_tracking import get_tracker
    
    clear()
    banner("Select Base Model")
    
    tracker = get_tracker()
    completed_models = tracker.get_completed_recon_models()
    
    if not completed_models:
        print(Fore.RED + "No completed reconstruction models found!")
        print(Fore.YELLOW + "Please train reconstruction models first.")
        input("\nPress ENTER to continue...")
        return None
    
    print(Fore.YELLOW + "Available Reconstruction Models:\n")
    
    for i, model in enumerate(completed_models, 1):
        config = model.get('config', {})
        print(Fore.GREEN + f" [{i}] {model['name']}")
        print(Fore.WHITE + f"     Pattern: {config.get('pattern', 'N/A')}")
        print(Fore.WHITE + f"     Seed: {config.get('seed', 'N/A')}")
        print(Fore.WHITE + f"     Timesteps: {config.get('timesteps', 'N/A')}")
        print(Fore.WHITE + f"     Status: {model.get('status', 'unknown')}")
        print()
    
    print(Fore.RED + " [0] Back\n")
    
    choice = input(Fore.WHITE + "Select model number: ").strip()
    
    if choice == "0":
        return None
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(completed_models):
            return completed_models[idx]
        else:
            print(Fore.RED + "Invalid selection!")
            input("Press ENTER...")
            return None
    except ValueError:
        print(Fore.RED + "Invalid input!")
        input("Press ENTER...")
        return None



