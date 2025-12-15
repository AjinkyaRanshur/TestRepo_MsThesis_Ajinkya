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
    banner("How do you want run these experiments")
    print(Fore.YELLOW + "Job Running Options:\n")
    print(Fore.GREEN + " [1] Slurm Submmission (Multiple Experiments)")
    print(Fore.GREEN + " [2] Interactive Running")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")


def slurm_entries():
    """
    Collect parameters for batch SLURM submission
    Returns a dictionary with all configuration parameters
    """
    clear()
    banner("BATCH SUBMISSION")
    
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



