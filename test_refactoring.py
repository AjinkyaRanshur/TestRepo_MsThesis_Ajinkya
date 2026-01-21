"""
Test script to verify refactored modules work correctly.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Test imports
print("="*60)
print("Testing imports...")
print("="*60)

try:
    from constants import (
        DEFAULT_TIMESTEPS,
        NOISE_LEVELS,
        TESTING_PATTERNS,
        get_device,
    )
    print("✓ constants.py imports successful")
except Exception as e:
    print(f"✗ constants.py import failed: {e}")
    sys.exit(1)

try:
    from checkpoint_utils import (
        initialize_feature_tensors,
        save_model_checkpoint,
        load_model_checkpoint,
    )
    print("✓ checkpoint_utils.py imports successful")
except Exception as e:
    print(f"✗ checkpoint_utils.py import failed: {e}")
    sys.exit(1)

try:
    from pattern_manager import PatternManager
    print("✓ pattern_manager.py imports successful")
except Exception as e:
    print(f"✗ pattern_manager.py import failed: {e}")
    sys.exit(1)

try:
    from dataset_manager import DatasetManager
    print("✓ dataset_manager.py imports successful")
except Exception as e:
    print(f"✗ dataset_manager.py import failed: {e}")
    sys.exit(1)

try:
    from metrics import (
        compute_classification_accuracy,
        compute_per_class_accuracy,
    )
    print("✓ metrics.py imports successful")
except Exception as e:
    print(f"✗ metrics.py import failed: {e}")
    sys.exit(1)

try:
    from network_refactored import Net
    print("✓ network_refactored.py imports successful")
except Exception as e:
    print(f"✗ network_refactored.py import failed: {e}")
    sys.exit(1)

try:
    from trainer import PredictiveCodingTrainer
    print("✓ trainer.py imports successful")
except Exception as e:
    print(f"✗ trainer.py import failed: {e}")
    sys.exit(1)

try:
    from test_runner import TestRunner
    print("✓ test_runner.py imports successful")
except Exception as e:
    print(f"✗ test_runner.py import failed: {e}")
    sys.exit(1)

# Test functionality
print("\n" + "="*60)
print("Testing functionality...")
print("="*60)

device = get_device()
print(f"✓ Device: {device}")

# Test pattern manager
print("\nTesting PatternManager...")
pm = PatternManager()
patterns = pm.get_available_patterns()
print(f"✓ Available patterns: {len(patterns)} patterns")
print(f"  Patterns: {list(patterns)[:3]}...")

pattern = pm.get_pattern("Uniform")
print(f"✓ Retrieved pattern 'Uniform': gamma={pattern['gamma']}")

# Test adding custom pattern
pm.add_pattern("Test", gamma=[0.1, 0.2, 0.3, 0.4], beta=[0.4, 0.3, 0.2, 0.1])
print(f"✓ Added custom pattern 'Test'")

# Test dataset manager
print("\nTesting DatasetManager...")
dm = DatasetManager()
datasets = dm.get_available_datasets()
print(f"✓ Available datasets: {datasets}")

classes = dm.get_all_patterns() if hasattr(dm, 'get_all_patterns') else dm.get_basic_shape_classes()
print(f"✓ Basic shape classes: {dm.get_basic_shape_classes()}")

# Test network
print("\nTesting Network...")
net = Net(num_classes=8, input_size=128)
net.to(device)
print(f"✓ Network created: {net.input_size}x{net.input_size}, {8} classes")

# Test feature tensor initialization
print("\nTesting feature tensor initialization...")
batch_size = 4
ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG = initialize_feature_tensors(
    batch_size=batch_size,
    height=128,
    width=128,
    device=device,
    include_dense=True,
)
print(f"✓ Initialized feature tensors for batch_size={batch_size}")
print(f"  ft_AB shape: {ft_AB.shape}")
print(f"  ft_BC shape: {ft_BC.shape}")
print(f"  ft_EF shape: {ft_EF.shape}")

# Test checkpoint saving/loading
print("\nTesting checkpoint operations...")
checkpoint_path = "/tmp/test_checkpoint.pt"
save_model_checkpoint(net, checkpoint_path, include_dense=True)
print(f"✓ Checkpoint saved to {checkpoint_path}")

net2 = Net(num_classes=8, input_size=128)
net2.to(device)
load_model_checkpoint(net2, checkpoint_path, device, include_dense=True)
print(f"✓ Checkpoint loaded")

# Test metrics
print("\nTesting metrics...")
predictions = torch.randn(10, 8)
targets = torch.randint(0, 8, (10,))
accuracy = compute_classification_accuracy(predictions, targets)
print(f"✓ Classification accuracy: {accuracy:.4f}")

# Test network forward pass
print("\nTesting network forward pass...")
dummy_input = torch.randn(batch_size, 3, 128, 128).to(device)
with torch.no_grad():
    output, ft_AB_t, ft_BC_t, ft_CD_t, ft_DE_t, ft_EF_t, recon = net.feedforward_pass(
        dummy_input, ft_AB, ft_BC, ft_CD, ft_DE
    )
print(f"✓ Feedforward pass successful")
print(f"  Output shape: {output.shape}")

# Test constants
print("\nTesting constants...")
print(f"✓ DEFAULT_TIMESTEPS: {DEFAULT_TIMESTEPS}")
print(f"✓ NOISE_LEVELS: {NOISE_LEVELS[:3]}... ({len(NOISE_LEVELS)} total)")
print(f"✓ Available patterns: {len(TESTING_PATTERNS)}")

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
