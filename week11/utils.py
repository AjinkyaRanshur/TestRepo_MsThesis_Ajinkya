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
    Simplified model naming convention
    
    Reconstruction: recon_t{timesteps}_{dataset}_{pattern}_s{seed}
    Classification: class_t{timesteps}_{base}_chk{epoch}_{pattern}_s{seed}
    
    Examples:
        recon_t10_c10_uni_s42
        class_t10_recon_t10_c10_uni_s42_chk150_uni_s42
    """
    # Pattern abbreviations
    pattern_map = {
        "Uniform": "uni",
        "Gamma Increasing": "ginc",
        "Gamma Decreasing": "gdec",
        "Beta Increasing": "binc",
        "Beta Decreasing": "bdec",
        "Beta Inc & Gamma Dec": "bgmix"
    }
    
    # Dataset abbreviations
    dataset_map = {
        "cifar10": "c10",
        "stl10": "stl",
        "custom_illusion_dataset": "ill"
    }
    
    p = pattern_map.get(pattern, pattern[:4].lower())
    
    if train_cond == "recon_pc_train":
        d = dataset_map.get(dataset, "unk")
        return f"recon_t{recon_timesteps}_{d}_{p}_s{seed}"
    
    elif train_cond == "classification_training_shapes":
        # Extract checkpoint epoch from base_model name
        if "_chk" in base_model:
            base_clean, chk_part = base_model.rsplit("_chk", 1)
            chk_epoch = chk_part
        else:
            base_clean = base_model
            chk_epoch = "0"
        
        return f"class_t{classification_timesteps}_{base_clean}_chk{chk_epoch}_{p}_s{seed}"
    
    return f"model_{p}_t{recon_timesteps}_s{seed}"
