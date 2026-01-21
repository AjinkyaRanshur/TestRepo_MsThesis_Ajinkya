# Refactoring Guide: Simplified Week11 Codebase

## Overview

The week11 codebase has been refactored to improve clarity, reduce redundancy, and make it easy to extend with new patterns and datasets. This guide explains the new structure and how to use it.

## Old Architecture Issues (Fixed)

- ✗ 20+ duplicates of feature tensor initialization
- ✗ 8+ duplicates of checkpoint loading code
- ✗ 680-line monolithic `eval_and_plotting.py`
- ✗ 220-line `train_test_loader()` function
- ✗ Hardcoded constants scattered throughout files
- ✗ Pattern logic mixed into training/testing code
- ✗ Dataset loading tightly coupled to training

## New Modular Structure

### Core Utilities
- **`constants.py`**: All hardcoded values, patterns, and configurations
- **`checkpoint_utils.py`**: Feature tensor initialization, checkpoint loading/saving
- **`metrics.py`**: Evaluation metrics (accuracy, loss, per-class metrics)
- **`pattern_manager.py`**: Extensible pattern system for testing and training
- **`dataset_manager.py`**: Extensible dataset system for easy addition of new datasets

### Training & Testing
- **`network_refactored.py`**: Refactored network with better documentation
- **`trainer.py`**: Unified training interface (replaces `illusion_pc_train.py` + `recon_pc_train.py`)
- **`test_runner.py`**: Unified test runner (consolidates trajectory, pattern, grid search testing)

### Legacy (Still Used For Now)
- **`add_noise.py`**: Image noise utility (unchanged)
- **`customdataset.py`**: Dataset loader (unchanged)

---

## Adding a New Pattern

### Simple Method: Register in `constants.py`

```python
# In constants.py - add to TESTING_PATTERNS dict
TESTING_PATTERNS = {
    "Uniform": {...},
    "My Custom Pattern": {
        "gamma": [0.1, 0.3, 0.5, 0.7],  # One value per layer
        "beta": [0.7, 0.5, 0.3, 0.1],
    },
}
```

### Programmatic Method: Use `PatternManager`

```python
from pattern_manager import PatternManager

pm = PatternManager()

# Add custom pattern
pm.add_pattern(
    "My Pattern",
    gamma=[0.1, 0.3, 0.5, 0.7],
    beta=[0.7, 0.5, 0.3, 0.1],
)

# Use in testing
pattern = pm.get_pattern("My Pattern")
print(pattern["gamma"])  # [0.1, 0.3, 0.5, 0.7]

# Create grid of patterns for experiments
patterns = PatternManager.create_custom_pattern_set(
    [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)],
    pattern_type="my_sweep"
)
```

---

## Adding a New Dataset

### Method 1: Register in `DatasetManager`

```python
from dataset_manager import DatasetManager

dm = DatasetManager()

# Register your dataset
dm.register_dataset(
    "my_dataset",
    {
        "type": "MyCustomDataset",  # Your class name
        "classes": ["class1", "class2", "class3"],
        "description": "My custom dataset",
    },
)

# Load with standard transforms
dataset = dm.load_dataset(
    "my_dataset",
    csv_path="data/my_dataset/metadata.csv",
    img_dir="data/my_dataset/images",
    transform=DatasetManager.get_standard_transforms(),
)
```

### Method 2: Create Custom Dataset Class

```python
# In dataset_manager.py, update load_dataset()
def load_dataset(self, name, csv_path, img_dir, ...):
    if dataset_type == "MyCustomDataset":
        from my_dataset import MyCustomDataset
        return MyCustomDataset(csv_path, img_dir, classes, transform)
```

---

## Training with New System

### Basic Usage

```python
import torch
from torch.utils.data import DataLoader
from network_refactored import Net
from trainer import PredictiveCodingTrainer
from dataset_manager import DatasetManager
from pattern_manager import PatternManager

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(num_classes=8, input_size=128)
net.to(device)

# Load dataset
dm = DatasetManager()
dataset = dm.load_dataset(
    "illusion",
    csv_path="data/visual_illusion_dataset/metadata.csv",
    img_dir="data/visual_illusion_dataset",
    transform=dm.get_standard_transforms(image_size=128),
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Setup training
optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)
trainer = PredictiveCodingTrainer(net, device, optimizer,
                                  training_condition="illusion_pc_train")

# Get pattern
pm = PatternManager()
pattern = pm.get_pattern("Uniform")

# Train
metrics = trainer.train_epoch(
    dataloader,
    timesteps=10,
    gammaset=[pattern["gamma"]],
    betaset=[pattern["beta"]],
    alphaset=[pattern["alpha"]],
)

print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
```

---

## Testing with New System

### Trajectory Testing

```python
from test_runner import TestRunner

test_runner = TestRunner(net, device)

# Test accuracy across timesteps 1-10
trajectory = test_runner.run_trajectory_test(
    test_dataloader,
    timesteps=10,
    patterns={"gamma": [0.33, 0.33, 0.33, 0.33],
              "beta": [0.33, 0.33, 0.33, 0.33],
              "alpha": [0.01, 0.01, 0.01, 0.01]},
)

# Results: {1: 0.45, 2: 0.55, 3: 0.62, ..., 10: 0.78}
print(trajectory)
```

### Pattern Testing

```python
# Test multiple patterns
pattern_results = test_runner.run_pattern_test(
    test_dataloader,
    pattern_names=["Uniform", "Gamma Increasing", "Beta Decreasing"],
    timesteps=10,
)

# Results: {"Uniform": 0.78, "Gamma Increasing": 0.82, ...}
print(pattern_results)
```

### Grid Search

```python
import numpy as np

# Search gamma and beta ranges
grid_results = test_runner.run_grid_search_test(
    test_dataloader,
    gamma_range=np.linspace(0.1, 0.9, 5),
    beta_range=np.linspace(0.1, 0.9, 5),
    timesteps=10,
)

# Results: {(0.1, 0.1): 0.45, (0.1, 0.3): 0.52, ...}
print(grid_results)
```

---

## Configuration Management

### Using Constants

```python
from constants import (
    DEFAULT_TIMESTEPS,
    NOISE_LEVELS,
    ILLUSION_DATASET_CLASSES,
    ALL_ILLUSION_CLASSES,
    TESTING_PATTERNS,
)

# Access built-in values
print(DEFAULT_TIMESTEPS)  # 10
print(NOISE_LEVELS)  # [0.0, 0.05, 0.1, ..., 0.35]
print(ALL_ILLUSION_CLASSES)  # ["square", "rectangle", ..., "all_in", "all_out"]
```

### Checkpoint Operations

```python
from checkpoint_utils import (
    initialize_feature_tensors,
    save_model_checkpoint,
    load_model_checkpoint,
)

# Initialize features for any batch size and image size
ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG = initialize_feature_tensors(
    batch_size=32,
    height=128,
    width=128,
    device=device,
    include_dense=True,
)

# Save checkpoint
save_model_checkpoint(net, "models/mymodel.pt", include_dense=True)

# Load checkpoint
load_model_checkpoint(net, "models/mymodel.pt", device, include_dense=True)
```

---

## Migration Path (Old → New)

### Replacing Old Training Code

**Old**: `illusion_pc_train.py` + `recon_pc_train.py`
**New**: `trainer.py` with `training_condition` parameter

```python
# Old way
import illusion_pc_train
illusion_pc_train.train_classification(...)  # 200+ lines

# New way
from trainer import PredictiveCodingTrainer
trainer = PredictiveCodingTrainer(net, device, optimizer,
                                  training_condition="illusion_pc_train")
metrics = trainer.train_epoch(dataloader, ...)
```

### Replacing Old Testing Code

**Old**: `test_workflow.py`, `pattern_testing.py`, `grid_search_testing.py`
**New**: `test_runner.py` with unified interface

```python
# Old way - 3 different testing functions
from test_workflow import run_trajectory_test
from pattern_testing import run_pattern_test
from grid_search_testing import run_grid_search

# New way - single class
from test_runner import TestRunner
runner = TestRunner(net, device)
runner.run_trajectory_test(...)
runner.run_pattern_test(...)
runner.run_grid_search_test(...)
```

---

## Benefits Summary

| Issue | Solution |
|-------|----------|
| Duplicate code (20+) | Centralized utilities in `checkpoint_utils.py` |
| Hardcoded values | All in `constants.py`, easy to modify |
| Difficult to add patterns | `PatternManager` with registration system |
| Difficult to add datasets | `DatasetManager` with registration system |
| Monolithic files (680+ LOC) | Split into focused modules with single responsibility |
| Testing scattered across 3 files | Unified `TestRunner` class |
| Training logic repeated | Unified `Trainer` class with configuration |
| Hard to understand network | `network_refactored.py` with docstrings and cleaner code |

---

## File Summary

| File | Purpose | LOC |
|------|---------|-----|
| `constants.py` | Global constants and defaults | ~130 |
| `checkpoint_utils.py` | Checkpoint and feature tensor utilities | ~200 |
| `metrics.py` | Evaluation metrics computation | ~150 |
| `pattern_manager.py` | Pattern registration and management | ~120 |
| `dataset_manager.py` | Dataset registration and loading | ~180 |
| `network_refactored.py` | Network architecture (refactored) | ~400 |
| `trainer.py` | Unified training interface | ~300 |
| `test_runner.py` | Unified testing interface | ~280 |

**Total new modules: ~1,760 LOC** (vs. original ~6,000+ LOC across 20+ files)

---

## Next Steps

1. Replace `network.py` with `network_refactored.py`
2. Update `main.py` to use new modules
3. Test compatibility with existing checkpoints
4. Gradually migrate other files to use new utilities
5. Remove redundant old files

