# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis project investigating how neural networks perceive visual illusions using predictive coding algorithms. The system trains feedforward-feedback neural networks on illusion datasets and tests whether model perception (measured via increasing softmax probability of ground-truth perception) aligns with human perception during predictive coding iterations.

**Core Research Question**: How do models trained on visual illusions learn hierarchical representations? Does PC dynamics reveal that models converge to human-like perception?

## Common Development Tasks

### Train a Model
```bash
cd week11
python interface.py
# [1] SLURM Job Submission → [1] Training Job → Select training mode
# Or direct: python main.py --config configs/config_0.py --model-name my_model
```

### Run Trajectory Testing
Tests whether model's **softmax probability of the perceived class increases across timesteps**:
```bash
python run_trajectory_test.py --models model_name1 model_name2 --timesteps 10 --dataset custom_illusion_dataset
# For illusion classes (all_in, all_out): tracks P(should_see) probability over time
# For shape classes: tracks P(class_name) probability over time
```

### Test on Different Datasets
```bash
# Custom illusion dataset (128×128, 6 basic shapes + 2 illusions = 8 classes)
python run_trajectory_test.py --models my_model --timesteps 10 --dataset custom_illusion_dataset

# Kanizsa dataset (32×32, 2 binary classes: square vs non-square based on should_see)
python run_trajectory_test.py --models my_model --timesteps 10 --dataset kanizsa_square_dataset

# Standard benchmarks (CIFAR-10, STL-10, etc.)
python main.py --config configs/config_0.py  # Set Dataset="cifar10" in config
```

### View Model Registry
```bash
python interface.py
# [2] View Model Registry
# Filter by training status, dataset, or patterns used
```

## Key Architectural Concepts

### Critical: Illusion Testing Logic (test_workflow.py:159-195)
This is the core mechanism for measuring model perception:

```python
# For each batch, during trajectory testing:
for i, cls_name in enumerate(cls_names):
    if cls_name in ["all_in", "all_out"]:
        # For illusions: track probability of should_see ground truth
        perceived_class = should_see[i]  # E.g., "square"
    else:
        # For basic shapes: track probability of the shape itself
        perceived_class = cls_name

    # Get model's softmax probability of perceived class
    perceived_idx = class_to_idx[perceived_class]
    class_results[cls_name]["predictions"][t].append(probs[i, perceived_idx])
```

**Key insight**: For `all_in` illusions (should perceive square), we check if P(square) increases over time. For `all_out` (should NOT perceive square), we also check if P(square) increases—if it doesn't, the model aligns with ground truth.

**This logic must not change**: It's the foundation of the research comparison between model and human perception.

## Datasets: Specifications & Peculiarities

### 1. Custom Illusion Dataset (128×128)
**Classes**: 6 basic shapes + 2 illusion classes = 8 total
- **Basic shapes** (6): square, rectangle, trapezium, triangle, hexagon, random
- **Illusions** (2): all_in, all_out
  - `all_in`: pacmen pointing inward → **should perceive "square"** (should_see = "square")
  - `all_out`: pacmen pointing outward → **should NOT perceive "square"** (should_see = "non-square" or similar)
- **Metadata file**: `dataset_metadata.csv` with columns: filename, Class, Should_See, Size, Position X, Position Y, Background Color, Shape Color
- **Usage**: Classification with noise augmentation (0-0.35 Gaussian noise in steps)
- **Test logic**: For illusions, check if P(should_see) increases; for shapes, check if P(class_name) increases

### 2. Kanizsa Square Dataset (32×32 - Binary)
**Classes**: 2 (based on should_see perception, not visual appearance)
- Class 1: `square` (Kanizsa square illusion - should perceive square)
- Class 2: `non-square` (random pacmen orientations - should NOT perceive square)
- **Key note**: Both classes are pacmen; classification is based on should_see ground truth, not shape
- **All images have**: should_see = "square" in metadata
- **Usage**: Binary classification, simpler PC dynamics for testing, 32×32 reduces computational cost

### 3. Other Available Datasets
- **CIFAR-10**: 32×32, 10 classes (reconstruction training baseline)
- **STL-10**: 96×96, 10 classes (higher-resolution baseline)
- New datasets can be added to `dataset_manager.py` (see "Extending for Other Datasets" section)

## Environment & Setup

**Environment File**: `environment.yml` defines conda environment named `cuda_pyt`
- PyTorch 2.5.1 with CUDA 11.8 support
- Key dependencies: numpy, matplotlib, tensorboard, torchvision, torchaudio, PyQt6
- All development should happen in `week11/` directory

**Setup Commands**:
```bash
conda env create -f environment.yml
conda activate cuda_pyt
cd week11
```

## Network Architecture

The `Net` class in `network.py` implements bidirectional predictive coding:
- **Feedforward** (Conv layers): 6→16→32→128 channels for hierarchical feature extraction
- **Feedback** (ConvTranspose layers): Reconstruct predictions for layer-wise error computation
- **Feature tensors** (ft_AB, ft_BC, ft_CD, ft_DE): Maintained during PC iterations for error signal propagation
- **Supports**: 32×32 (CIFAR-10, Kanizsa) and 128×128 (custom illusions)

## Training Modes

1. **recon_pc_train**: Reconstruction loss (learn to predict images from noise)
2. **illusion_pc_train**: Classification loss on illusions with noise augmentation (0-0.35 Gaussian)
3. **fine_tuning**: Transfer learning (currently planned, not implemented)

## File Organization & Consolidation Opportunities

> **Note**: The following consolidations maintain full functionality while reducing navigation complexity and code duplication. These are recommendations for future refactoring.

### Priority 1: Config Consolidation (Highest Impact)
**Problem**: `config_0.py` through `config_11.py` are 95% identical (~1,100 bytes each), creating massive redundancy

**Current**: 12 config files + base_config.py + configfile.py + configtest.py = 15 files
**Recommendation**:
- Create `configs/experiments.json` registry with all experiment parameters as data
- Create `configs/config_loader.py` with ConfigLoader and ConfigBuilder classes
- Delete 12 redundant `config_*.py` files
- Delete stale `configfile.py` and `configtest.py`

**Impact**: Removes 14 files, reduces ~13 KB of redundant Python, centralizes experiment tracking

### Priority 2: Test Consolidation
**Problem**: 7 test-related files with ~40% duplicated logic

**Current**: test_workflow.py + test_runner.py + run_trajectory_test.py + run_pattern_test.py + run_grid_search_test.py + pattern_testing.py + grid_search_testing.py
**Recommendation**:
- Merge `run_trajectory_test.py`, `run_pattern_test.py`, `run_grid_search_test.py` into single `run_test.py` dispatcher
- Consolidate pattern_testing.py and grid_search_testing.py logic into test_workflow.py
- Deprecate test_runner.py (duplicate of test_workflow.py)

**Impact**: Reduces 5-7 files → 2 files, eliminates TestConfig duplication, single entry point for all tests

### Priority 3: Training Consolidation
**Problem**: `illusion_pc_train.py` and `recon_pc_train.py` are 70% duplicate code

**Current**: illusion_pc_train.py + recon_pc_train.py + trainer.py + main.py orchestration
**Recommendation**:
- Deprecate `illusion_pc_train.py` and `recon_pc_train.py`
- Use unified `trainer.py` (PredictiveCodingTrainer class) as sole training interface
- Update main.py to only use trainer.py

**Impact**: Removes 537 lines of redundant training logic, single code path for maintenance

### Priority 4: Dataset Consolidation
**Problem**: Dataset generators scattered; duplicated geometry logic

**Current**: customdataset.py + dataset_manager.py + Sequential_Dataset_Generation.py + data/kanisza_square_dataset.py + data/shapes_dataset.py
**Recommendation**: Create `data_generation/` package:
```
data_generation/
  ├── __init__.py
  ├── geometry.py           # Shared pacman/shape drawing utilities
  ├── base.py               # Abstract dataset base class
  └── generators/
      ├── illusion.py       # Custom illusions (from Sequential_Dataset_Generation.py)
      └── kanisza.py        # Merged kanisza_square_dataset.py + shapes_dataset.py
```

**Impact**: ~200 lines of deduplicated geometry code, better organization, single source of truth for drawing logic

## Extending for Other Datasets (PC Dynamics)

To use predictive coding dynamics on custom datasets (CIFAR-10, STL-10, medical imaging, etc.):

### Step 1: Create Custom Dataset Class
```python
# In data_generation/generators/my_dataset.py
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __getitem__(self, idx):
        # CRITICAL: Return 4-tuple with consistent format:
        # (image, label, class_name, should_see)
        # where should_see is ground truth perception (or None if not applicable)
        return image, label, class_name, should_see
```

### Step 2: Register Dataset
```python
# In dataset_manager.py, update load_dataset():
if dataset_name == "my_dataset":
    num_classes = 10
    train_dataset = MyDataset(...)
    val_dataset = MyDataset(...)
    return train_dataset, val_dataset, num_classes
```

### Step 3: Configure Training
```python
# In configs/my_config.py
Dataset = "my_dataset"
classification_datasetpath = "path/to/my/dataset"
number_of_classes = 10
```

### Step 4: Adapt Testing Logic
Modify `test_workflow.py:159-195` to handle your domain-specific perception logic:
```python
# Example: For medical imaging classification
if dataset == "my_medical_dataset":
    # Custom logic: maybe should_see is diagnosis ground truth
    perceived_class = should_see[i] if should_see[i] else cls_name
else:
    # existing illusion logic stays unchanged
```

## Implementation Details

### Feature Tensor Management (Critical)
During each PC iteration, feature tensors are:
- **Initialized** to zeros at batch dimensions
- **Maintained** with gradients for error signal computation
- **Cleaned up** after PC pass to prevent memory leaks (see test_workflow.py:197-198)

### Model Checkpoint Format
```python
checkpoint = {
    "conv1": net.conv1.state_dict(),  # All conv layers
    "conv2": ...,
    "deconv1_fb": net.deconv1_fb.state_dict(),  # All deconv layers
    # ... etc for all layers
}
```

### Reproducibility
- Config `seed` parameter controls all randomness sources
- SLURM jobs lock specific timesteps, models, seeds for deterministic testing
- Model registry tracks config used for each trained model

## Debugging & Troubleshooting

**Testing logic**: Check test_workflow.py:159-195 (where perceived_class is selected)
**Dataset loading**: Run test_dataloader.py to verify dataset structure
**PC convergence**: Monitor feature tensor gradients, check noise levels
**Memory**: Reduce batch_size, verify torch.cuda.empty_cache() in loops
**Config path**: Paths are relative to week11/, use absolute paths if needed
