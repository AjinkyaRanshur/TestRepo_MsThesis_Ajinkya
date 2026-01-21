# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis project investigating neural networks trained on visual illusions using predictive coding algorithms. The codebase implements feedforward-feedback neural networks with reconstruction and classification tasks applied to illusion datasets.

**Key Research Area**: How models trained on visual illusions learn hierarchical representations, with emphasis on model perception vs. ground-truth perception of illusions.

## Repository Structure

**Core Training & Architecture**:
- `network.py`: Neural network with Conv layers (feedforward pathway) and ConvTranspose layers (feedback pathway) for predictive coding
- `illusion_pc_train.py`: Training loop for illusion classification with predictive coding
- `recon_pc_train.py`: Training loop for reconstruction tasks
- `customdataset.py`: Dataset loader for visual illusion dataset (SquareDataset class)

**Experiment Management**:
- `week11/configs/`: Configuration files for experiments (base_config.py contains parameters like learning rate, batch size, timesteps, noise parameters)
- `model_tracking.py`: Tracks trained models with metadata (uses model_registry.json)
- `model_registry.json`: JSON registry of all trained models and their configurations

**Testing & Evaluation**:
- `test_workflow.py`: Unified testing interface supporting trajectory testing, pattern testing, and grid search
- `run_trajectory_test.py`: Standalone script for trajectory testing (model accuracy across timesteps)
- `run_pattern_test.py`: Pattern formation testing
- `run_grid_search_test.py`: Grid search over hyperparameters
- `eval_and_plotting.py`: Evaluation metrics and visualization

**Utilities & Data**:
- `add_noise.py`: Adds Gaussian or salt-and-pepper noise to images
- `Sequential_Dataset_Generation.py`: Generates visual illusion datasets
- `main.py`: Main training orchestration (imports config and coordinates training)
- `menu_options.py`: Interactive menu system for experiment selection
- `utils.py`: Helper utilities (clear screen, banners, etc.)
- `batch_submissions.py`: Creates SLURM batch job scripts
- `interface.py`: GUI interface for interactive experimentation

**Data Structure**:
- `week{1-11}/`: Weekly progress folders with evolving code
- `data/visual_illusion_dataset/`: Main illusion dataset with metadata CSV and class subdirectories
- `models/`: Saved model checkpoints
- `plots/`: Generated visualizations
- `slurm_jobs/`: SLURM job submissions and logs
- `runs/`: TensorBoard/WandB experiment logs

## Environment & Setup

**Environment File**: `environment.yml` defines conda environment named `cuda_pyt`
- PyTorch 2.5.1 with CUDA 11.8 support
- Key dependencies: numpy, matplotlib, tensorboard, torchvision, torchaudio
- Includes PyQt6 for GUI

**Setup Commands**:
```bash
conda env create -f environment.yml
conda activate cuda_pyt
```

## Development & Training Commands

**Single Training Run**:
```bash
# Navigate to a week folder and run main.py with a config
cd week11
python main.py
# main.py dynamically imports config from the configs/ folder
```

**Configuration System**:
- Config is a Python module in `configs/` folder (e.g., `configs/config_0.py`)
- Each config defines: batch_size, epochs, lr, timesteps, noise_type, noise_param, device, dataset paths, model name, etc.
- Training mode determined by `training_condition` in config: "recon_pc_train", "illusion_pc_train", or "fine_tuning"
- Layers to optimize controlled by `optimize_all_layers` flag in config

**Running Tests**:
```bash
# Trajectory testing (model accuracy across timesteps)
python run_trajectory_test.py --models model_name1 model_name2 --timesteps 10 --dataset custom_illusion_dataset

# Pattern testing
python run_pattern_test.py

# Grid search
python run_grid_search_test.py
```

**SLURM Cluster Submission**:
```bash
# Create and submit SLURM job for testing
python slurm_testing_submission.py
# Creates scripts in slurm_jobs/ and submits via sbatch
```

## Key Concepts & Architecture

### Predictive Coding Network Architecture
The `Net` class in `network.py` implements a bidirectional network:
- **Feedforward pathway** (Conv layers): Extracts hierarchical features (6 → 16 → 32 → 128 channels)
- **Feedback pathway** (ConvTranspose layers): Reconstructs image from predictions for error computation
- **Features stored at each layer**: Temporary feature tensors (ft_AB, ft_BC, ft_CD, ft_DE, etc.) propagated during predictive coding iterations
- Input size adaptive: supports 32x32 (CIFAR-10) and 128x128 (STL-10) images

### Dataset Structure
Visual illusion dataset consists of 8 classes:
- **Basic shapes** (6 classes): square, rectangle, trapezium, triangle, hexagon, random
- **Illusion classes** (2 classes): all_in, all_out
- **Metadata**: CSV with filename, class, Should_See ground truth (whether the illusion should be perceived)
- **Data splits**: 70% train (basic + random), 30% validation, test set includes matched samples from illusion classes

### Configuration Management
Configs are Python modules with experiment parameters:
- `base_config.py`: Template configuration
- `config_0.py`, `config_1.py`, etc.: Specific experiment configs
- Main config attributes: `batch_size`, `epochs`, `lr`, `timesteps`, `seed`, `device`, `classification_neurons`, `noise_type`, `noise_param`, `model_name`, `training_condition`, `optimize_all_layers`

### Model Tracking
- Models stored with metadata in `model_registry.json`
- Each model entry contains: model name, config parameters, training date, performance metrics
- Enables querying models by name or configuration for reproducible testing

### Training Modes
1. **illusion_pc_train**: Classification on illusion dataset with noise augmentation (0-0.35 Gaussian noise in steps)
2. **recon_pc_train**: Reconstruction task with reconstruction loss
3. **fine_tuning**: Supervised classification with optional layer selection (all layers vs. linear layers only)

## Testing & Metrics

**Trajectory Testing**: Tests how model performance evolves across timesteps (T=1 to T=10)
- Aggregates results across multiple model seeds
- Returns accuracy per class with error bars
- Key function: `run_trajectory_test()` in `test_workflow.py`

**Pattern Testing**: Evaluates model response to specific geometric patterns
**Grid Search**: Sweeps hyperparameters (learning rate, noise, timesteps) to find optimal configurations

**Evaluation Metrics**:
- Classification accuracy per class and overall
- Reconstruction loss (MSE between reconstruction and original)
- Per-timestep accuracy tracking
- "Illusion index": Metric comparing model perception to ground-truth Should_See

## Important Notes

### Data Paths
- Dataset paths configured in config files, typically:
  - Illusion dataset: `data/visual_illusion_dataset/`
  - External datasets (CIFAR-10, STL-10): configured in config via `recon_datasetpath`
  - Model save/load: paths set in config (`save_model_path`, `load_model_path`)

### CUDA/Device Handling
- Network configured for CUDA if available
- Device passed through config: `config.device`
- Batch data and model moved to device within training loops
- Check device availability: `torch.cuda.is_available()`

### Week Folders as Iterations
- Code evolves across `week1/` through `week11/` folders
- Each week may refactor implementation, add new features, or change directory structure
- Most active development in `week11/` (latest implementation)
- `week10/` and earlier are historical references but may have outdated patterns

### Reproducibility
- Set seed via `set_seed(seed)` function in main.py
- Config includes seed parameter for all randomness sources (Python random, numpy, torch, CUDA)
- SLURM jobs configure specific timesteps, models, and seeds for deterministic testing
