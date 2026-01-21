"""
Simplified training orchestration using refactored modules.

This demonstrates how to use the new modular architecture:
- constants.py for configuration
- pattern_manager.py for patterns
- dataset_manager.py for datasets
- trainer.py for training
- checkpoint_utils.py for checkpoints
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import sys
import argparse
import random
import numpy as np

from constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_TIMESTEPS,
    ILLUSION_DATASET_CLASSES,
    ALL_ILLUSION_CLASSES,
    TRAIN_SPLIT,
    VAL_SPLIT,
    get_device,
)
from pattern_manager import PatternManager
from dataset_manager import DatasetManager
from network import Net
from trainer import PredictiveCodingTrainer
from checkpoint_utils import save_model_checkpoint, load_model_checkpoint


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_split_dataset(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    image_size: int = 128,
    train_split_ratio: float = TRAIN_SPLIT,
) -> tuple:
    """
    Load dataset and split into train/val/test.

    Args:
        dataset_name: Name of dataset to load
        data_dir: Path to dataset directory
        batch_size: Batch size for dataloaders
        image_size: Target image size
        train_split_ratio: Ratio for train/val split

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
    """
    dm = DatasetManager()

    # Get transforms
    transforms_composed = DatasetManager.get_standard_transforms(
        image_size=image_size,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    # Load dataset
    csv_path = os.path.join(data_dir, "dataset_metadata.csv")
    dataset = dm.load_dataset(
        dataset_name,
        csv_path=csv_path,
        img_dir=data_dir,
        subset_classes=ALL_ILLUSION_CLASSES,
        transform=transforms_composed,
    )

    print(f"Loaded dataset: {len(dataset)} samples")

    # Split into train/val (test will use validation set with illusion matches)
    train_size = int(len(dataset) * train_split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(DEFAULT_SEED),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = val_loader  # Use val as test for now

    num_classes = len(ALL_ILLUSION_CLASSES)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")
    print(f"Num classes: {num_classes}")

    return train_loader, val_loader, test_loader, num_classes


def train_model(
    config: dict,
) -> None:
    """
    Train model using unified trainer.

    Args:
        config: Configuration dictionary with:
            - model_name: Name for the model
            - num_classes: Number of output classes
            - input_size: Input image size (32 or 128)
            - batch_size: Batch size
            - epochs: Number of epochs
            - lr: Learning rate
            - timesteps: Predictive coding timesteps
            - training_condition: Type of training
            - data_dir: Path to dataset
            - checkpoint_path: Path to save checkpoints
    """
    print("\n" + "="*60)
    print(f"Training Configuration")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    # Setup
    device = get_device()
    print(f"Device: {device}\n")

    set_seed(config.get("seed", DEFAULT_SEED))

    # Load data
    train_loader, val_loader, test_loader, num_classes = load_and_split_dataset(
        dataset_name="illusion",
        data_dir=config.get("data_dir", "data/visual_illusion_dataset"),
        batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
        image_size=config.get("input_size", 128),
    )

    # Create network
    net = Net(
        num_classes=num_classes,
        input_size=config.get("input_size", 128),
    )
    net.to(device)
    print(f"Network created with {num_classes} classes")

    # Setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=config.get("lr", DEFAULT_LR))
    print(f"Optimizer: Adam, LR: {config.get('lr', DEFAULT_LR)}\n")

    # Create trainer
    trainer = PredictiveCodingTrainer(
        net=net,
        device=device,
        optimizer=optimizer,
        training_condition=config.get("training_condition", "illusion_pc_train"),
    )

    # Get patterns
    pm = PatternManager()
    pattern_name = config.get("pattern", "Uniform")
    pattern = pm.get_pattern(pattern_name)
    print(f"Using pattern: {pattern_name}")
    print(f"  Gamma: {pattern['gamma']}")
    print(f"  Beta: {pattern['beta']}\n")

    # Training loop
    best_val_acc = 0.0
    checkpoint_path = config.get("checkpoint_path", "models/best_model.pt")
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    print("="*60)
    print("Starting Training...")
    print("="*60 + "\n")

    for epoch in range(config.get("epochs", DEFAULT_EPOCHS)):
        # Train
        train_metrics = trainer.train_epoch(
            dataloader=train_loader,
            timesteps=config.get("timesteps", DEFAULT_TIMESTEPS),
            gammaset=[pattern["gamma"]],
            betaset=[pattern["beta"]],
            alphaset=[pattern["alpha"]],
        )

        # Validate
        val_metrics = trainer.evaluate(
            dataloader=val_loader,
            timesteps=config.get("timesteps", DEFAULT_TIMESTEPS),
            gammaset=[pattern["gamma"]],
            betaset=[pattern["beta"]],
            alphaset=[pattern["alpha"]],
        )

        val_acc = val_metrics["eval_accuracy"]

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_checkpoint(
                net,
                checkpoint_path,
                include_dense=True,
                metadata={
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "pattern": pattern_name,
                },
            )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{config.get('epochs', DEFAULT_EPOCHS)}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Acc: {val_metrics['eval_accuracy']:.4f}")
            print(f"  Best Val Acc: {best_val_acc:.4f}\n")

    print("="*60)
    print(f"Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train predictive coding network")
    parser.add_argument(
        "--model_name",
        type=str,
        default="pc_illusion_model",
        help="Name of the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Predictive coding timesteps",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="Uniform",
        help="Pattern to use (Uniform, Gamma Increasing, etc.)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=128,
        help="Input image size",
    )
    parser.add_argument(
        "--training_condition",
        type=str,
        default="illusion_pc_train",
        help="Type of training (illusion_pc_train or recon_pc_train)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/visual_illusion_dataset",
        help="Path to dataset",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/best_model.pt",
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )

    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "timesteps": args.timesteps,
        "pattern": args.pattern,
        "input_size": args.input_size,
        "training_condition": args.training_condition,
        "data_dir": args.data_dir,
        "checkpoint_path": args.checkpoint_path,
        "seed": args.seed,
        "num_classes": 8,  # Fixed for illusion dataset
    }

    train_model(config)


if __name__ == "__main__":
    main()
