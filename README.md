# Predictive Coding Neural Networks on Visual Illusions

A research implementation investigating how neural networks trained on visual illusions learn hierarchical representations using predictive coding algorithms.

**Status**: Refactored, optimized, and ready for training/testing
**Branch**: `claude/init-project-setup-TQpEb`

---

## Table of Contents

- [Quick Start](#quick-start)
- [Setup & Installation](#setup--installation)
- [Project Overview](#project-overview)
- [Training Models](#training-models)
  - [Interactive Menu (Windows/Linux)](#interactive-menu-windowslinux)
  - [SLURM Cluster Submission (Linux)](#slurm-cluster-submission-linux)
  - [Direct Training](#direct-training)
- [Testing & Evaluation](#testing--evaluation)
- [Available Datasets](#available-datasets)
- [Available Patterns](#available-patterns)
- [Project Structure](#project-structure)
- [Configuration System](#configuration-system)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate cuda_pyt
cd week11
```

### 2. Interactive Mode (Recommended for First-Time Users)
```bash
# Start interactive menu - works on Windows, Linux, and WSL
python interface.py
```

### 3. Check Available Models
```bash
# View all trained models and their status
python interface.py
# Select: [2] View Model Registry
```

---

## Setup & Installation

### Requirements
- Python 3.12+
- CUDA 11.8 (for GPU training)
- 32GB RAM recommended
- ~60GB disk space (for datasets)

### Installation Steps

```bash
# 1. Create conda environment from file
conda env create -f environment.yml
conda activate cuda_pyt

# 2. Navigate to main code directory
cd week11

# 3. Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Platform-Specific Notes

**Linux/WSL**: All features supported (interactive menu + SLURM)

**Windows**: Interactive menu supported via `interface.py` (no SLURM support)

**macOS**: Interactive menu supported (no CUDA, CPU-only mode)

---

## Project Overview

### Research Goal
Understand how neural networks perceive visual illusions by training them on illusion datasets using predictive coding algorithms.

### Key Architecture
- **Feedforward pathway**: Conv layers for hierarchical feature extraction
- **Feedback pathway**: ConvTranspose layers for reconstruction/prediction
- **Predictive coding**: Iterative refinement using feature-level error signals

### Training Modes
1. **Reconstruction Training** (`recon_pc_train`): Learn to reconstruct images with noise
2. **Classification Training** (`classification_training_shapes`): Classify shapes vs. illusions
3. **Fine-tuning** (upcoming): Transfer learning on classification

---

## Training Models

### Option 1: Interactive Menu (Recommended)

The interactive menu provides a user-friendly interface for model training with options to configure all hyperparameters.

```bash
cd week11
python interface.py
```

**Menu Flow**:
```
Main Menu
├── [1] SLURM Job Submission
│   ├── [1] Training Job
│   │   ├── [1] Reconstruction Training
│   │   └── [2] Classification Training
│   └── [2] Testing Job
│       ├── [1] Trajectory Testing
│       ├── [2] Pattern Testing
│       └── [3] Grid Search Testing
├── [2] View Model Registry
├── [3] Interactive Mode
├── [4] Generate Aggregate Plots
└── [0] Exit
```

#### Example: Reconstruction Training via Menu
```
1. Select: [1] SLURM Job Submission
2. Select: [1] Training Job
3. Select: [1] Reconstruction Training
4. Enter parameters:
   - Epochs: 200
   - Batch size: 40
   - Learning rate: 0.00005
   - Timesteps: 10
   - Dataset: cifar10 or custom_illusion_dataset
   - Patterns: Select from [Uniform, Gamma Increasing, ...]
   - Seeds: 3
5. Confirm submission
```

**Cross-Platform Compatibility**:
- ✅ Linux/WSL: Full support
- ✅ Windows: Full support via `python interface.py`
- ✅ macOS: Full support (CPU mode)

---

### Option 2: SLURM Cluster Submission (Linux/WSL Only)

For GPU clusters, submit batch jobs directly via SLURM.

#### Reconstruction Training (SLURM)
```bash
cd week11
python interface.py
# Select: [1] SLURM Job Submission → [1] Training → [1] Reconstruction
# Fill in parameters and confirm
```

**Auto-generated SLURM Script Features**:
- Runs 2 jobs in parallel (on 2 GPUs)
- Automatically saves checkpoints every 10 epochs
- Generates aggregate plots after completion
- Logs stored in `slurm_jobs/` directory

#### Classification Training (SLURM)
```bash
cd week11
python interface.py
# Select: [1] SLURM Job Submission → [1] Training → [2] Classification
# Select base reconstruction models
# Fill in parameters
```

**Requirements**:
- `sbatch` command available (SLURM cluster)
- 2 GPUs minimum recommended
- 32GB+ RAM

---

### Option 3: Direct Training

For manual control or debugging:

```bash
cd week11

# Reconstruction training
python main.py --config configs/config_0.py --model-name my_recon_model

# View main.py arguments
python main.py --help
```

---

## Testing & Evaluation

### Test Types

#### 1. Trajectory Testing
Measures model accuracy across timesteps (T=1 to T=10).

```bash
cd week11
python interface.py
# Select: [1] SLURM Job Submission → [2] Testing Job → [1] Trajectory Testing
```

Or direct:
```bash
python run_trajectory_test.py --models model_name1 model_name2 --timesteps 10
```

#### 2. Pattern Testing
Evaluates model response to different predictive coding patterns.

```bash
python interface.py
# Select: [1] SLURM Job Submission → [2] Testing Job → [2] Pattern Testing
```

Available patterns:
- Uniform
- Gamma Increasing
- Gamma Decreasing
- Beta Increasing
- Beta Decreasing
- Beta Inc & Gamma Dec

#### 3. Grid Search
Sweeps hyperparameters to find optimal configurations.

```bash
python interface.py
# Select: [1] SLURM Job Submission → [2] Testing Job → [3] Grid Search Testing
```

### Generate Aggregate Plots
After training completes:

```bash
python interface.py
# Select: [4] Generate Aggregate Plots
# Choose: [1] Aggregate all completed model groups
```

---

## Available Datasets

### 1. Custom Illusion Dataset (Recommended)
**Path**: `data/visual_illusion_dataset/`
- **Size**: 54 MB (92,160 images)
- **Classes**: 8 (6 basic shapes + 2 illusion classes)
  - Basic shapes: square, rectangle, trapezium, triangle, hexagon, random
  - Illusions: all_in, all_out
- **Split**: 70% train, 30% validation
- **Format**: PNG images with metadata CSV

**Metadata**: `dataset_metadata.csv`
```csv
filename,class,should_see
shape_001.png,square,1
shape_002.png,square,1
illusion_001.png,all_in,0
```

### 2. Kanizsa Square Dataset
**Path**: `data/kanizsa_square_dataset/`
- **Size**: 1.6 MB (2,952 images)
- **Classes**: 2 (Kanizsa square vs. random)
- **Usage**: Binary classification testing

### 3. CIFAR-10
- **Size**: Auto-downloaded (~170 MB)
- **Classes**: 10 (standard benchmark)
- **Usage**: Reconstruction training baseline

### 4. STL-10
- **Size**: Auto-downloaded (~2.6 GB)
- **Classes**: 10 (ImageNet subset)
- **Usage**: High-resolution reconstruction training

**Select during training**:
```
Available datasets:
  1. cifar10 (32×32 images)
  2. stl10 (96×96 images)
  3. custom_illusion_dataset (128×128 images)
  4. kanizsa_square_dataset (128×128 binary)
```

---

## Available Patterns

Predictive coding patterns control how feature tensors are initialized during training iterations.

### Pattern Library

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Uniform** | Equal weight to all timesteps | Baseline/comparison |
| **Gamma Increasing** | Weight increases over timesteps | Temporal dynamics testing |
| **Gamma Decreasing** | Weight decreases over timesteps | Early convergence testing |
| **Beta Increasing** | Alternative increasing schedule | Alternative dynamics |
| **Beta Decreasing** | Alternative decreasing schedule | Alternative dynamics |
| **Beta Inc & Gamma Dec** | Combined pattern | Complex dynamics |

### Pattern Effects on Training

Each pattern modulates the strength of error signals across iterations:
- Increasing patterns: Later timesteps have more influence
- Decreasing patterns: Early timesteps have more influence
- Combined patterns: Complex temporal dynamics

### Adding Custom Patterns

Edit `week11/constants.py`:
```python
PATTERNS = {
    "Uniform": {"gamma": [0.3]*10, "beta": [0.3]*10},
    "MyPattern": {"gamma": [0.1, 0.2, 0.3, ...], "beta": [...]},
}
```

---

## Project Structure

```
week11/
├── Training & Architecture
│   ├── network.py                  # Neural network (Conv + ConvTranspose)
│   ├── illusion_pc_train.py        # Classification training loop
│   ├── recon_pc_train.py           # Reconstruction training loop
│   ├── main.py                     # Training orchestration
│   └── customdataset.py            # Dataset loader
│
├── Testing & Evaluation
│   ├── test_workflow.py            # Unified testing interface
│   ├── run_trajectory_test.py      # Trajectory testing script
│   ├── run_pattern_test.py         # Pattern testing script
│   ├── run_grid_search_test.py     # Hyperparameter grid search
│   ├── test_runner.py              # Test execution framework
│   ├── test_dataloader.py          # Dataset testing
│   └── test_refactoring.py         # Validation tests
│
├── Configuration & Management
│   ├── configs/
│   │   ├── base_config.py          # Config template
│   │   ├── config_0.py to config_11.py  # Experiment configs
│   │   └── generated/              # Auto-generated configs
│   ├── constants.py                # Centralized constants
│   ├── create_config.py            # Dynamic config generator
│   ├── model_tracking.py           # Model registry management
│   └── model_registry.json         # Model metadata (auto-managed)
│
├── Data & Utilities
│   ├── data/
│   │   ├── visual_illusion_dataset/     # 92K illusion images
│   │   ├── kanizsa_square_dataset/      # 3K binary images
│   │   ├── shapes_dataset.py            # Dataset generator
│   │   └── kanisza_square_dataset.py    # Kanizsa loader
│   ├── utils.py                    # Helper utilities
│   ├── add_noise.py                # Noise augmentation
│   ├── checkpoint_utils.py         # Checkpoint I/O
│   ├── metrics.py                  # Evaluation metrics
│   ├── pattern_manager.py          # Pattern system
│   ├── dataset_manager.py          # Dataset system
│   └── post_training_aggregation.py    # Results aggregation
│
├── Interactive Interface
│   ├── interface.py                # Main UI loop
│   ├── menu_options.py             # Menu functions
│   ├── batch_submissions.py        # SLURM script generation
│   └── slurm_testing_submission.py # Test job generation
│
├── Documentation
│   ├── README.md                   # This file
│   ├── CLAUDE.md                   # Claude Code guidance
│   ├── README_REFACTORED.md        # Refactoring details
│   └── REFACTORING_GUIDE.md        # API documentation
│
├── Output Directories (Auto-created)
│   ├── models/                     # Model checkpoints
│   ├── plots/                      # Visualizations
│   │   ├── individual_training_metrics/
│   │   ├── training_metrics/
│   │   └── test_trajectories/
│   └── slurm_jobs/                 # SLURM scripts & logs
│
└── Supporting Files
    ├── environment.yml             # Conda environment
    ├── network_old.py              # Original implementation
    └── archived_files_backup.tar.gz # Cached files backup
```

---

## Configuration System

### Configuration Files

Configs are Python modules in `configs/` defining training hyperparameters.

### Base Configuration Template
```python
# base_config.py - ALL available parameters

batch_size = 40
epochs = 200
lr = 0.00005
timesteps = 10
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
Dataset = "custom_illusion_dataset"
data_path = "data/visual_illusion_dataset/"
recon_datasetpath = "data/cifar10/"

# Model
model_name = "recon_t10_c40_lr5e5_s42"
save_model_path = "models/recon_t10_c40_lr5e5_s42.pth"

# Patterns
pattern = "Uniform"
gamma_start, gamma_stop = 0.3, 0.3
beta_start, beta_stop = 0.3, 0.3

# Noise (classification only)
noise_type = "gaussian"
noise_param = 0.1

# Training mode
training_condition = "recon_pc_train"
optimize_all_layers = True

# Classification
classification_neurons = 128
number_of_classes = 10
```

### Creating Custom Configs

**Option 1: Via Interface**
```bash
python interface.py
# Menu automatically generates configs based on your input
```

**Option 2: Manual**
```bash
# Copy and modify a template
cp week11/configs/base_config.py week11/configs/my_config.py
# Edit my_config.py with your parameters
python main.py --config configs/my_config.py
```

### Key Parameters Explained

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `batch_size` | 40 | 16-128 | Larger = faster but more memory |
| `lr` | 0.00005 | 1e-6 to 1e-3 | Lower for more stable training |
| `timesteps` | 10 | 1-100 | More = slower but potentially better convergence |
| `epochs` | 200 | 10-500 | Classification: 25, Recon: 200 |
| `pattern` | "Uniform" | See Available Patterns | Controls PC dynamics |
| `noise_param` | 0.1 | 0.0-0.35 | Gaussian std or SP probability |

---

## Model Registry

The system tracks all trained models in `model_registry.json`.

### Viewing Registry

```bash
python interface.py
# Select: [2] View Model Registry
# Filter by: All / Reconstruction / Classification / Completed
```

### Registry Entry Example
```json
{
  "name": "recon_t10_c40_lr5e5_s1234",
  "config": {
    "batch_size": 40,
    "epochs": 200,
    "lr": 0.00005,
    "timesteps": 10,
    "seed": 1234,
    "pattern": "Uniform",
    "Dataset": "custom_illusion_dataset"
  },
  "status": "completed",
  "created_at": "2026-01-21T10:30:00",
  "training_started": "2026-01-21T10:31:00",
  "training_completed": "2026-01-21T14:45:00",
  "metrics": {
    "final_loss": 0.0234,
    "best_loss": 0.0198
  },
  "checkpoint_path": "models/recon_t10_c40_lr5e5_s1234.pth"
}
```

### Model Status
- `pending`: Registered but not started
- `training`: Currently training
- `submitted`: Submitted to SLURM
- `completed`: Finished successfully
- `failed`: Training failed

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce `batch_size` (try 16-24)
- Use `--device cpu` for debugging
- Check GPU availability: `nvidia-smi`

#### 2. Dataset Not Found
```
FileNotFoundError: data/visual_illusion_dataset/ not found
```
**Solution**:
```bash
# Dataset should be in week11/data/
# Verify path:
ls -la week11/data/visual_illusion_dataset/
# Download if needed (see Data section)
```

#### 3. Model Not Found During Testing
```
KeyError: model_name not in registry
```
**Solution**:
- Train a model first or check registry
- View available models: `python interface.py` → [2]

#### 4. SLURM Job Failed
```
Check logs: week11/slurm_jobs/*.err
```
**Common causes**:
- Missing conda environment: `conda activate cuda_pyt`
- Missing GPU: `sinfo -N | grep gpu`
- Registry lock: Wait for previous job to complete

#### 5. Aggregate Plots Not Generated
```
python post_training_aggregation.py
```
**Verify**:
- Check `plots/` directory exists
- Models must have `completed` status
- At least 2 models in same group

### Debug Mode

```bash
# Run with verbose output
python main.py --config configs/config_0.py --verbose

# Test data loading only
python test_dataloader.py

# Validate configuration
python -c "from configs.config_0 import *; print(f'Device: {device}, Batch: {batch_size}')"
```

---

## Training Examples

### Example 1: Quick Reconstruction Training

**Goal**: Train a reconstruction model for 50 epochs (5 min for quick testing)

```bash
cd week11
python interface.py
# [1] SLURM Job Submission → [1] Training → [1] Reconstruction
# Epochs: 50 (faster)
# Batch size: 40 (default)
# Learning rate: 0.00005 (default)
# Timesteps: 10 (default)
# Dataset: cifar10 (smaller, faster)
# Patterns: Uniform (simplest)
# Seeds: 1 (single seed for speed)
# Proceed
```

**Expected Output**:
```
✓ Created SLURM script: slurm_jobs/pred_net_recon_pc_train_20260121_103000.sh
✓ Generated 1 config files
✓ Registered 1 models in tracker
```

### Example 2: Comprehensive Classification Study

**Goal**: Train classification models on multiple patterns with 3 seeds

```bash
cd week11
python interface.py
# [1] SLURM Job Submission → [1] Training → [2] Classification
# Select base models: [1] (first reconstruction model)
# Epochs: 25 (classification default)
# Batch size: 40
# Learning rate: 0.00005
# Timesteps: 10
# Patterns: 1,2,3 (Uniform, Gamma Inc, Gamma Dec)
# Seeds: 3
# Optimizer: [1] Linear layers only (faster)
# Proceed
```

**Expected**: 9 models trained (3 patterns × 3 seeds)

### Example 3: Direct Training with Custom Config

```bash
# Create custom config
cat > week11/configs/my_custom.py << 'EOF'
from base_config import *

batch_size = 16
epochs = 100
lr = 0.0001
timesteps = 5
model_name = "my_custom_model"
pattern = "Gamma Increasing"
EOF

# Train
cd week11
python main.py --config configs/my_custom.py --model-name my_custom_model

# Results in: models/my_custom_model.pth
```

---

## Performance Benchmarks

### Single Model Training Time (GPU)
| Config | Time | VRAM |
|--------|------|------|
| Recon 50 epochs, CIFAR-10 | ~2-3 min | 6 GB |
| Recon 200 epochs, Illusion | ~15-20 min | 8 GB |
| Classification 25 epochs | ~3-5 min | 7 GB |
| Test (trajectory, 10 models) | ~30-60 sec | 4 GB |

### Batch Training (SLURM, 2 GPUs)
| Config | Models | Time |
|--------|--------|------|
| 2 recon × 3 patterns | 6 | ~60 min |
| 1 recon × 4 patterns × 3 seeds × 2 checkpoints | 24 | ~180 min |
| Testing (trajectory) | ~50 | ~15 min (parallel) |

---

## Documentation

### Available Guides
- **README.md** (this file): User guide and examples
- **CLAUDE.md**: Architecture and Claude Code guidance
- **README_REFACTORED.md**: Refactoring details and API overview
- **REFACTORING_GUIDE.md**: Detailed API documentation
- **Code docstrings**: Inline documentation in each module

### Getting Help

1. **Quick questions**: Check README sections above
2. **API details**: See REFACTORING_GUIDE.md
3. **Architecture**: See CLAUDE.md
4. **Code examples**: See bottom of test files
5. **Debugging**: See Troubleshooting section

---

## Contributing & Extending

### Add a New Pattern
1. Edit `week11/constants.py`
2. Add pattern to `PATTERNS` dict
3. Use via interface: "Select patterns → Custom"

### Add a New Dataset
1. Create loader in `week11/data/`
2. Register in `dataset_manager.py`
3. Select during training setup

### Add Evaluation Metrics
1. Add function to `week11/metrics.py`
2. Update test runners to use new metric
3. Generate plots with new metric

---

## Citation

If you use this code, please cite:
```bibtex
@thesis{ajinkya2026thesis,
  title={Predictive Coding Neural Networks on Visual Illusions},
  author={Ajinkya, [Your Name]},
  year={2026},
  school={[Your Institution]}
}
```

---

## License

[Specify your license - e.g., MIT, Apache 2.0, etc.]

---

## Questions or Issues?

For questions about this codebase:
1. Check documentation files listed above
2. Review configuration examples
3. Check model registry for training status
4. Review SLURM logs: `week11/slurm_jobs/*.err`

**For bugs**: Create an issue with error logs and configuration used

---

**Last Updated**: January 21, 2026
**Version**: 1.0 (Refactored & Optimized)
**Status**: Ready for Training & Testing
