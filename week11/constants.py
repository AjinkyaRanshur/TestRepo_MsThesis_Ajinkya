"""
Global constants and configuration values for the predictive coding framework.
Centralizes all hardcoded values for easy modification and consistency.
"""

import torch

# Network Architecture
NETWORK_ARCHITECTURE = {
    "conv_channels": [6, 16, 32, 128],
    "kernel_size": 5,
    "num_fc_layers": 3,
    "fc_hidden_sizes": [1024, 256],
}

# Input size to FC input size mapping (after 4 pooling layers)
INPUT_SIZE_TO_FC = {
    32: 128 * 2 * 2,    # 32 -> 16 -> 8 -> 4 -> 2
    128: 128 * 8 * 8,   # 128 -> 64 -> 32 -> 16 -> 8
}

# Training Parameters
DEFAULT_BATCH_SIZE = 40
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.00005
DEFAULT_SEED = 42
DEFAULT_TIMESTEPS = 10

# Noise Configuration
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
DEFAULT_NOISE_TYPE = "s&p"  # "gaussian" or "s&p" (salt-and-pepper)
DEFAULT_NOISE_PARAM = 0.0

# Predictive Coding Parameters
DEFAULT_GAMMA_SET = [[0.33, 0.33, 0.33, 0.33]]  # Uniform pattern
DEFAULT_BETA_SET = [[0.33, 0.33, 0.33, 0.33]]
DEFAULT_ALPHA_SET = [[0.01, 0.01, 0.01, 0.01]]

# Dataset Configuration
ILLUSION_DATASET_CLASSES = {
    "basic_shapes": ["square", "rectangle", "trapezium", "triangle", "hexagon"],
    "illusions": ["all_in", "all_out"],
    "other": ["random"],
}

ALL_ILLUSION_CLASSES = (
    ILLUSION_DATASET_CLASSES["basic_shapes"]
    + ILLUSION_DATASET_CLASSES["illusions"]
    + ILLUSION_DATASET_CLASSES["other"]
)

# Data Split Ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.3

# Predefined Testing Patterns
TESTING_PATTERNS = {
    "Uniform": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33],
    },
    "Gamma Increasing": {
        "gamma": [0.13, 0.33, 0.53, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33],
    },
    "Gamma Decreasing": {
        "gamma": [0.53, 0.33, 0.13, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33],
    },
    "Beta Increasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33],
    },
    "Beta Decreasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.53, 0.33, 0.13, 0.33],
    },
    "Mixed": {
        "gamma": [0.13, 0.33, 0.53, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33],
    },
}

# Training Conditions
TRAINING_CONDITIONS = {
    "recon_pc_train": "Reconstruction with predictive coding",
    "illusion_pc_train": "Classification on illusions with predictive coding",
    "fine_tuning": "Fine-tuning transfer learning from reconstruction model",
}

# Device Configuration
def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
