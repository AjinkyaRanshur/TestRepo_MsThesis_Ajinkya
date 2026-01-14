import os
import pyfiglet
from colorama import Fore, Style, init


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def banner(text):
    print(Fore.CYAN + pyfiglet.figlet_format(text, font="ogre"))

def parse_list(x, cast=int):
    return [cast(v.strip()) for v in x.split(",")]

def generate_model_name(pattern, seed, train_cond, recon_timesteps, 
                       classification_timesteps=None, dataset=None, base_model=None):
    """
    FIX 3: Simplified and clearer model naming
    
    Reconstruction: recon_{dataset}_{pattern}_t{timesteps}_s{seed}
    Classification: class_{base_recon_name}_{pattern}_t{timesteps}_s{seed}
    """
    # Shorten pattern names for cleaner filenames
    pattern_abbrev = {
        "Uniform": "uni",
        "Gamma Increasing": "ginc",
        "Gamma Decreasing": "gdec",
        "Beta Increasing": "binc",
        "Beta Decreasing": "bdec",
        "Beta Inc & Gamma Dec": "bgmix"
    }
    
    # Shorten dataset names
    dataset_abbrev = {
        "cifar10": "c10",
        "stl10": "stl",
        "custom_illusion_dataset": "illusion"
    }
    
    p = pattern_abbrev.get(pattern, pattern.lower()[:4])
    
    if train_cond == "recon_pc_train":
        d = dataset_abbrev.get(dataset, dataset[:3] if dataset else "unk")
        return f"recon_{d}_{p}_t{recon_timesteps}_s{seed}"
    
    elif train_cond == "classification_training_shapes":
        # Extract just the base recon model name without the checkpoint info
        if base_model and "_chk" in base_model:
            base_name = base_model.split("_chk")[0]
            chk = base_model.split("_chk")[1]
        else:
            base_name = base_model if base_model else "unknown"
            chk = "0"
        
        return f"class_{base_name}_chk{chk}_{p}_t{classification_timesteps}_s{seed}"
    
    return f"model_{p}_t{recon_timesteps}_s{seed}"
