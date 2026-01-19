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
                       classification_timesteps=None, dataset=None, base_model=None,
                       optimize_all_layers=False):
    """
    Simplified model naming convention
    
    Reconstruction: recon_t{timesteps}_{dataset}_{pattern}_s{seed}
    Classification: class_t{timesteps}_{base}_chk{epoch}_{pattern}_{opt}_s{seed}
    
    NEW: Added optimizer scope differentiation
    - _lin = linear layers only
    - _all = all layers (conv + linear)
    
    Examples:
        recon_t10_c10_uni_s42
        class_t10_recon_t10_c10_uni_s42_chk150_uni_lin_s42  (linear only)
        class_t10_recon_t10_c10_uni_s42_chk150_uni_all_s42  (all layers)
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
        
        # NEW: Add optimizer scope to name
        opt_suffix = "all" if optimize_all_layers else "lin"
        
        return f"class_t{classification_timesteps}_{base_clean}_chk{chk_epoch}_{p}_{opt_suffix}_s{seed}"
    
    return f"model_{p}_t{recon_timesteps}_s{seed}"


def find_seed_siblings(model_name):
    """
    Find all models with the same configuration but different seeds
    UPDATED: Now considers optimize_all_layers for classification models
    
    Args:
        model_name: Name of the model to find siblings for
    
    Returns:
        List of model names (including the input model) that share the same config
    """
    from model_tracking import get_tracker
    
    tracker = get_tracker()
    
    # Get the model's config
    model_info = tracker.get_model(model_name)
    if not model_info:
        return [model_name]
    
    config = model_info['config']
    train_cond = config.get('train_cond')
    
    # Get all models of the same type
    all_models = tracker.get_models_by_type(train_cond)
    
    # Filter to only completed models
    all_models = [m for m in all_models if m.get('status') == 'completed']
    
    # Find siblings with matching config (excluding seed)
    siblings = []
    
    for model in all_models:
        model_config = model['config']
        
        # Check if configs match (excluding seed)
        if train_cond == "recon_pc_train":
            if (model_config.get('pattern') == config.get('pattern') and
                model_config.get('timesteps') == config.get('timesteps') and
                model_config.get('Dataset') == config.get('Dataset') and
                model_config.get('lr') == config.get('lr')):
                siblings.append(model['name'])
        
        elif train_cond == "classification_training_shapes":
            # UPDATED: Now includes optimize_all_layers in matching
            if (model_config.get('pattern') == config.get('pattern') and
                model_config.get('timesteps') == config.get('timesteps') and
                model_config.get('Dataset') == config.get('Dataset') and
                model_config.get('lr') == config.get('lr') and
                model_config.get('base_recon_model') == config.get('base_recon_model') and
                model_config.get('checkpoint_epoch') == config.get('checkpoint_epoch') and
                model_config.get('optimize_all_layers') == config.get('optimize_all_layers')):
                siblings.append(model['name'])
    
    return siblings


def extract_config_from_model_name(model_name):
    """
    Extract configuration info from model name for grouping
    UPDATED: Now includes optimize_all_layers for classification
    
    Returns:
        tuple: Configuration key for grouping models
    """
    from model_tracking import get_tracker
    
    tracker = get_tracker()
    model_info = tracker.get_model(model_name)
    
    if not model_info:
        return None
    
    config = model_info['config']
    train_cond = config.get('train_cond')
    
    if train_cond == "recon_pc_train":
        return (
            config.get('pattern'),
            config.get('timesteps'),
            config.get('Dataset'),
            config.get('lr')
        )
    elif train_cond == "classification_training_shapes":
        # UPDATED: Now includes optimize_all_layers in grouping key
        return (
            config.get('pattern'),
            config.get('timesteps'),
            config.get('Dataset'),
            config.get('lr'),
            config.get('base_recon_model'),
            config.get('checkpoint_epoch'),
            config.get('optimize_all_layers')
        )
    
    return None
