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
    clear()
    banner("BATCH SUBMISSION")
    
    print("\nTraining Conditions")
    training_condition = input("Training condition (recon_pc_train/classification_training_shapes): ").strip()
   
    print("\n Base Configrations,If Multiple Parameters Need to be tested please use a comma after Value:")
    epochs = int(input("Epochs(default 200): ") or "200")
    epochs = parse_list(epochs,int)
    batch_size = int(input("Batch size(default 40): ") or "40")
    batch_size = parse_list(batch_size,int)
    lr = float(input("Learning rate(default 0.00005): ") or "0.00005")
    lr=parse_list(lr,float)
    timesteps = int(input("Timesteps(default 10): ") or "10")
    timesteps = parse_list(timesteps,int)
    number_of_classes=int(input("Enter Number of classes 10 for CIFAR and 6 for Illusory dataset") or 6)
    base_config={
	"train_cond":training_condition,
	"epochs":epochs,
	"batch_size":batch_size,
	"lr":lr,
	"timesteps":timesteps,
	"number_of_classes":number_of_classes
    
    }
    
    
    # Select patterns
    print("\nAvailable patterns:")
    
    patterns = ["Uniform", "Gamma Increasing", "Gamma Decreasing", 
                "Beta Increasing", "Beta Decreasing"]
    
    for i, p in enumerate(patterns, 1):
        print(f"{i}. {p}")
    
    pattern_choice = input("Select patterns (comma-separated numbers, or 'all'): ").strip()
    
    if pattern_choice.lower() == 'all':
        selected_patterns = patterns
    else:
        indices = [int(x.strip())-1 for x in pattern_choice.split(',')]
        selected_patterns = [patterns[i] for i in indices]
    
    # Select seeds
    seed_input = input("\nEnter seeds (comma-separated, e.g., 42,123,456): ").strip()
    seeds = [int(x.strip()) for x in seed_input.split(',')]
       
    number_of_models=len{epochs} * len(batch_size) * len(lr) * len(timesteps) * len(seeds) * len(selected_patterns)

    print(f"\nWill create {number_of_models} models")
    
    base_config={
        "train_cond":training_condition,
        "epochs":epochs,
        "batch_size":batch_size,
        "lr":lr,
        "timesteps":timesteps,
        "number_of_classes":number_of_classes,
	"selected_patterns":selected_patterns,
	"seeds":seeds

    }
   
    
    return base_config



