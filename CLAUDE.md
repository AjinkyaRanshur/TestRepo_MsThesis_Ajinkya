# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pred Net for Psychosis Tasks**: Master's thesis project investigating how neural networks perceive visual illusions using predictive coding algorithms. The system trains feedforward-feedback neural networks on illusion datasets and tests whether model perception (measured via increasing softmax probability of ground-truth perception) aligns with human perception during predictive coding iterations.

**Core Research Question**: How do models trained on visual illusions learn hierarchical representations? Does PC dynamics reveal that models converge to human-like perception patterns relevant to psychosis research?

## Common Development Tasks

### Train a Model
```bash
python interface.py
# [1] SLURM Job Submission → [1] Training Job → Select training mode

# Or direct with experiment ID (0-11):
python main.py --config 0 --model-name my_model

# Or with explicit config path:
python main.py --config configs.base_config --model-name my_model
```

### Run Testing
```bash
# Trajectory testing (track P(perceived_class) over timesteps)
python run_test.py --mode trajectory --models model_name1 model_name2 --timesteps 10 --dataset custom_illusion_dataset

# Pattern testing (compare different gamma/beta patterns)
python run_test.py --mode pattern --models my_model --timesteps 10 --patterns "Uniform,Gamma Increasing"

# Grid search (sweep gamma/beta parameter space)
python run_test.py --mode grid --models my_model --timesteps 10 \
    --gamma-start 0.1 --gamma-stop 0.5 --gamma-step 0.1 \
    --beta-start 0.1 --beta-stop 0.5 --beta-step 0.1
```

### View Model Registry
```bash
python interface.py
# [2] View Model Registry
```

## Key Architectural Concepts

### Critical: Illusion Testing Logic (test_workflow.py:159-195)
This is the core mechanism for measuring model perception:

```python
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

**This logic must not change**: It's the foundation of the research comparison between model and human perception.

## Datasets

### Custom Illusion Dataset (128x128, 8 classes)
- **Basic shapes** (6): square, rectangle, trapezium, triangle, hexagon, random
- **Illusions** (2): all_in, all_out
  - `all_in`: pacmen pointing inward → should perceive "square"
  - `all_out`: pacmen pointing outward → should NOT perceive "square"

### Kanizsa Square Dataset (32x32, binary)
- Class 1: `square` (Kanizsa illusion - perceive square)
- Class 2: `non-square` (random pacmen - don't perceive square)

### Standard Benchmarks
- **CIFAR-10**: 32x32, 10 classes
- **STL-10**: 96x96, 10 classes

## File Structure

```
├── main.py                 # Main training orchestration
├── network.py              # Predictive coding network architecture
├── illusion_pc_train.py    # Classification training with PC
├── recon_pc_train.py       # Reconstruction training with PC
├── test_workflow.py        # Core testing logic
├── run_test.py             # Unified test runner CLI
├── pattern_testing.py      # Pattern-based testing
├── grid_search_testing.py  # Grid search testing
├── interface.py            # Interactive menu interface
├── configs/
│   ├── base_config.py      # Template config
│   └── experiments.py      # All experiment configs (replaces config_0-11.py)
├── data/
│   ├── kanisza_square_dataset.py
│   └── shapes_dataset.py
└── customdataset.py        # Dataset class
```

## Configuration System

Experiments are defined in `configs/experiments.py`. Use experiment IDs 0-11:

```python
# From code:
from configs.experiments import get_config
config = get_config(0)  # Get experiment 0 config

# From CLI:
python main.py --config 0
```

## Network Architecture

The `Net` class in `network.py` implements bidirectional predictive coding:
- **Feedforward**: 3→6→16→32→128 channels (Conv layers)
- **Feedback**: Mirrored ConvTranspose layers for reconstruction
- **Supports**: 32x32 and 128x128 input sizes

## Training Modes

1. **recon_pc_train**: Reconstruction loss (predict images from noise)
2. **illusion_pc_train**: Classification loss with noise augmentation

## Debugging

- **Testing logic**: Check test_workflow.py:159-195
- **Dataset loading**: Run test_dataloader.py
- **PC convergence**: Monitor feature tensor gradients
- **Memory**: Reduce batch_size, use torch.cuda.empty_cache()
