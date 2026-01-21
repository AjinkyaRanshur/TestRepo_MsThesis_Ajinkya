"""
Unified experiment configuration system.
Replaces individual config_*.py files with a single source of truth.

Usage:
    from configs.experiments import get_config
    config = get_config(0)  # or get_config("config_0")
"""

import torch
from types import SimpleNamespace


# Default configuration values
DEFAULTS = {
    "batch_size": 40,
    "epochs": 200,
    "lr": 5e-05,
    "momentum": 0.9,
    "classification_datasetpath": "custom_illusion_dataset",
    "recon_datasetpath": "custom_illusion_dataset",
    "training_condition": "recon_pc_train",
    "classification_neurons": 10,
    "timesteps": 10,
    "gammaset": [[0.33, 0.33, 0.33, 0.33]],
    "betaset": [[0.33, 0.33, 0.33, 0.33]],
    "alphaset": [[0.01, 0.01, 0.01, 0.01]],
    "noise_type": "s&p",
    "noise_param": 0.0,
    "load_model_path": "/home/ajinkyar/ml_models",
    "save_model_path": "/home/ajinkyar/ml_models",
}


# Experiment-specific overrides (only values that differ from defaults)
EXPERIMENTS = {
    0: {"seed": 8733, "model_name": "recon_t10_ill_uni_s8733"},
    1: {"seed": 5511, "model_name": "recon_t10_ill_uni_s5511"},
    2: {"seed": 4013, "model_name": "recon_t10_c10_uni_s4013",
        "recon_datasetpath": "cifar10"},
    3: {"seed": 4013, "model_name": "recon_t10_stl_uni_s4013",
        "recon_datasetpath": "stl10"},
    4: {"seed": 5234, "model_name": "recon_t10_ill_uni_s5234"},
    5: {"seed": 410, "model_name": "recon_t10_ill_uni_s410"},
    6: {"seed": 410, "model_name": "recon_t10_c10_uni_s410",
        "recon_datasetpath": "cifar10"},
    7: {"seed": 410, "model_name": "recon_t10_stl_uni_s410",
        "recon_datasetpath": "stl10"},
    8: {"seed": 4507, "model_name": "recon_t10_ill_uni_s4507"},
    9: {"seed": 4507, "model_name": "recon_t10_c10_uni_s4507",
        "recon_datasetpath": "cifar10"},
    10: {"seed": 4507, "model_name": "recon_t10_stl_uni_s4507",
         "recon_datasetpath": "stl10"},
    11: {"seed": 4013, "model_name": "recon_t10_ill_uni_s4013"},
}


def get_device():
    """Get CUDA device if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_config(experiment_id):
    """
    Get configuration for an experiment.

    Args:
        experiment_id: Integer ID (0-11) or string like "config_0"

    Returns:
        SimpleNamespace with all config attributes
    """
    # Parse experiment ID
    if isinstance(experiment_id, str):
        if experiment_id.startswith("configs."):
            experiment_id = experiment_id.replace("configs.", "")
        if experiment_id.startswith("config_"):
            experiment_id = int(experiment_id.replace("config_", ""))

    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment ID: {experiment_id}. Valid: 0-11")

    # Build config from defaults + experiment overrides
    config = dict(DEFAULTS)
    config.update(EXPERIMENTS[experiment_id])

    # Add computed fields
    config["device"] = get_device()
    config["experiment_name"] = (
        f"Testing {config['model_name']} with Uniform pattern at "
        f"{config['timesteps']} timesteps"
    )

    return SimpleNamespace(**config)


def print_device_info(device):
    """Print CUDA device information."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Using CUDA device: {device}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    print(f"Using device: {device}")


# For backward compatibility: expose as module-level variables when imported directly
if __name__ != "__main__":
    # Default to experiment 0 for backward compatibility
    _default = get_config(0)
    for key, value in vars(_default).items():
        globals()[key] = value
