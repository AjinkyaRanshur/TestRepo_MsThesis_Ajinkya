# Week 11 Testing Guide

## Overview
This guide explains how to run different types of tests on the week11 code for predictive coding models with visual illusions.

## Prerequisites

### Environment Setup
1. Install conda environment from `environment.yml`:
```bash
conda env create -f environment.yml
conda activate cuda_pyt
```

2. Or install required packages with pip:
```bash
pip install torch torchvision matplotlib numpy colorama pillow
```

### Model Requirements
- Trained models should be registered in `model_registry.json`
- Model checkpoints should be available at the paths specified in the registry
- Check model status: models can be "registered", "submitted", "training", or "completed"

## Validation Tests

### 1. Syntax Validation
Validates Python syntax without requiring dependencies:
```bash
cd week11
python validate_code.py
```

Expected output: All 23 Python files should pass syntax validation.

## Model Testing (Requires Trained Models)

### 2. Trajectory Testing
Tests model predictions across timesteps with standard patterns.

**Small test run:**
```bash
cd week11
python run_trajectory_test.py \
    --models recon_t10_ill_uni_s5511 \
    --timesteps 5 \
    --dataset custom_illusion_dataset
```

**Full test run:**
```bash
python run_trajectory_test.py \
    --models recon_t10_ill_uni_s5511 \
    --timesteps 10 \
    --dataset custom_illusion_dataset
```

**Multiple models:**
```bash
python run_trajectory_test.py \
    --models recon_t10_ill_uni_s5511 recon_t10_ill_uni_s5234 \
    --timesteps 10 \
    --dataset custom_illusion_dataset
```

### 3. Pattern Testing
Tests models with different gamma/beta patterns.

**Available patterns:**
- Uniform
- Gamma Increasing
- Gamma Decreasing
- Beta Increasing
- Beta Decreasing
- Beta Inc & Gamma Dec

**Small test run (2 patterns):**
```bash
cd week11
python run_pattern_test.py \
    --models recon_t10_ill_uni_s5511 \
    --timesteps 5 \
    --patterns "Uniform,Gamma Increasing" \
    --dataset custom_illusion_dataset
```

**Full test run (all patterns):**
```bash
python run_pattern_test.py \
    --models recon_t10_ill_uni_s5511 \
    --timesteps 10 \
    --patterns "Uniform,Gamma Increasing,Gamma Decreasing,Beta Increasing,Beta Decreasing,Beta Inc & Gamma Dec" \
    --dataset custom_illusion_dataset
```

### 4. Grid Search Testing
Tests models across a range of gamma and beta values.

**Small test run (narrow ranges):**
```bash
cd week11
python run_grid_search_test.py \
    --models recon_t10_ill_uni_s5511 \
    --timesteps 5 \
    --dataset custom_illusion_dataset \
    --gamma-start 0.2 \
    --gamma-stop 0.5 \
    --gamma-step 0.1 \
    --beta-start 0.2 \
    --beta-stop 0.5 \
    --beta-step 0.1
```

**Full test run (wide ranges):**
```bash
python run_grid_search_test.py \
    --models recon_t10_ill_uni_s5511 \
    --timesteps 10 \
    --dataset custom_illusion_dataset \
    --gamma-start 0.1 \
    --gamma-stop 0.6 \
    --gamma-step 0.05 \
    --beta-start 0.1 \
    --beta-stop 0.6 \
    --beta-step 0.05
```

## Available Datasets
- `custom_illusion_dataset` - Visual illusion dataset (128x128 images, 6 classes)
- `kanizsa_square_dataset` - Kanizsa square illusion dataset

## Model Registry

Check available models:
```python
import json
with open('model_registry.json') as f:
    registry = json.load(f)
    for name, info in registry['models'].items():
        print(f"{name}: {info['status']}")
```

Currently available completed models:
- `recon_t10_ill_uni_s5511` - Completed model trained on custom_illusion_dataset

## Output
All tests generate:
- Console output with progress and results
- Plots saved to `plots/` directory (for trajectory and pattern testing)
- Summary statistics printed at completion

## Testing Checklist

### Quick Validation (No Models Required)
- [ ] Run `python validate_code.py` - all files pass syntax check

### Small Test Runs (Fast, Requires 1 Model)
- [ ] Trajectory test with 5 timesteps
- [ ] Pattern test with 2 patterns
- [ ] Grid search with small ranges (3x3 grid)

### Full Test Runs (Slow, Requires Multiple Models)
- [ ] Trajectory test with 10 timesteps across all models
- [ ] Pattern test with all 6 patterns
- [ ] Grid search with full parameter space

## Troubleshooting

### No module named 'torch'
Install PyTorch: `pip install torch torchvision`

### Model not found
Check model is in `model_registry.json` and status is "completed"

### Checkpoint path not found
Verify the checkpoint_path in model_registry.json exists:
```python
import json
registry = json.load(open('model_registry.json'))
model_info = registry['models']['recon_t10_ill_uni_s5511']
print(model_info['checkpoint_path'])
```

### CUDA errors
Tests will automatically fall back to CPU if CUDA is unavailable.

## Notes
- Test duration depends on number of timesteps, models, and patterns
- Small test runs should complete in < 5 minutes
- Full test runs may take 30+ minutes depending on configuration
- All tests save plots to the `plots/` directory for visual inspection
