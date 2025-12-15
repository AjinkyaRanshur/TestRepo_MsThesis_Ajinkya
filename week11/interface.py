import os
import pyfiglet
from colorama import Fore, Style, init
import sys
from main import load_config, main
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from utils import clear,banner
from menu_options import slurm_entries,job_running,main_menu,train_menu,classification_train_menu,test_menu

# init colorama for Windows
init(autoreset=True)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG_FILE = "configs/configfile.py"
CONFIG_MODULE = "configfile"


# Pattern definitions
PATTERNS = {
    "Uniform": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Gamma Increasing": {
        "gamma": [0.13, 0.33, 0.53, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Gamma Decreasing": {
        "gamma": [0.53, 0.33, 0.13, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Beta Increasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33]
    },
    "Beta Decreasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.53, 0.33, 0.13, 0.33]
    },
    "Beta Inc & Gamma Dec": {
        "gamma": [0.53, 0.33, 0.13, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33]
    }
}



def run():
    while True:
        job_type = job_running()
        
        if job_type == "1":
            # SLURM submission
            base_config = slurm_entries()
            if base_config:
                from batch_submissions import create_slurm_script, submit_sbatch
                script_path, model_ids = create_slurm_script(base_config)
                
                confirm = input("\nSubmit to SLURM? (y/n): ").strip().lower()
                if confirm == 'y':
                    submitted_info = submit_sbatch(script_path, model_ids)
                    input("\nPress ENTER to continue...")
        
        elif job_type == "2":
            # Interactive running
            choice = main_menu()
            
            if choice == "1":
                # Training menu
                train_choice = train_menu()
                if train_choice == "1":
                    print("Starting reconstruction training...")
                    # Call your training function
                elif train_choice == "2":
                    print("Starting classification training...")
                    # Show available reconstruction models
                    from menu_options import model_selection_menu
                    base_model = model_selection_menu()
                    if base_model:
                        print(f"Selected model: {base_model['name']}")
                        # Call classification training
            
            elif choice == "2":
                # Testing menu
                test_choice = test_menu()
                # Implement testing options
            
            elif choice == "0":
                clear()
                banner("GoodBye!")
                break
            else:
                print(Fore.RED + "Invalid option!")
                input("Press ENTER...")
        
        elif job_type == "0":
            clear()
            banner("GoodBye!")
            break
        else:
            print(Fore.RED + "Invalid option!")
            input("Press ENTER...")



if __name__ == "__main__":
    run()























