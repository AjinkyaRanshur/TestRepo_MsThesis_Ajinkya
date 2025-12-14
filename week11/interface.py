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
from menu_options import slurm_entries,job_runnning,main_menu,train_menu,classification_train_menu,test_menu,classification_models

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
        job_type=job_running()
        if job_type == "1":
	   base_config=slurm_entries()
           something=create_slurm_script(base_config)
           submitted_info=sbatch_submission()

	elif job_type =="2":
         choice = main_menu()
	 if choice == "1":
	 elif choice == "2":
	 elif choice == "3":
	 elif choice == "0":
	    clear()
            banner("GoodBye!")
            break
	 else:
	     print(Fore.RED + "Invalid option!")
             input("Press ENTER...")



if __name__ == "__main__":
    run()























