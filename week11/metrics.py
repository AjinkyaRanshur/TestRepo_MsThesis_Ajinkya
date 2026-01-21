"""
Evaluation metrics for classification and reconstruction tasks.
Simplified from eval_and_plotting.py, focuses only on metric computation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from constants import NOISE_LEVELS


def compute_classification_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth targets

    Returns:
        Accuracy value (0-1)
    """
    if predictions.dim() > 1:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions

    correct = (pred_classes == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def compute_per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_labels: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute accuracy for each class.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        class_labels: Class names for each sample
        num_classes: Total number of classes

    Returns:
        Dictionary mapping class names to accuracies
    """
    if predictions.dim() > 1:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions

    accuracies = {}
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            correct = (pred_classes[class_mask] == targets[class_mask]).sum().item()
            total = class_mask.sum().item()
            accuracies[class_idx] = correct / total
        else:
            accuracies[class_idx] = 0.0

    return accuracies


def compute_reconstruction_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
) -> float:
    """
    Compute MSE reconstruction loss.

    Args:
        reconstruction: Reconstructed image
        original: Original image

    Returns:
        Loss value
    """
    loss = torch.nn.functional.mse_loss(reconstruction, original)
    return loss.item()


def compute_per_timestep_accuracy(
    all_timestep_predictions: List[torch.Tensor],
    targets: torch.Tensor,
) -> List[float]:
    """
    Compute accuracy for each timestep.

    Args:
        all_timestep_predictions: List of predictions for each timestep
        targets: Ground truth targets

    Returns:
        List of accuracies per timestep
    """
    accuracies = []
    for timestep_pred in all_timestep_predictions:
        acc = compute_classification_accuracy(timestep_pred, targets)
        accuracies.append(acc)
    return accuracies


def aggregate_accuracies_across_seeds(
    all_seed_accuracies: List[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate accuracies across multiple seeds (runs).

    Args:
        all_seed_accuracies: List of accuracy lists from different seeds

    Returns:
        Tuple of (mean accuracies, std accuracies)
    """
    accuracies = np.array(all_seed_accuracies)
    mean_acc = np.mean(accuracies, axis=0)
    std_acc = np.std(accuracies, axis=0)
    return mean_acc, std_acc


def compute_illusion_index(
    model_perception: np.ndarray,
    ground_truth_should_see: np.ndarray,
) -> Dict[str, float]:
    """
    Compute illusion perception metrics.

    Compares what the model perceives vs. ground truth "should_see" labels.

    Args:
        model_perception: Model predictions for illusions (1 = perceived, 0 = not)
        ground_truth_should_see: Ground truth should_see labels

    Returns:
        Dictionary with perception metrics
    """
    correct_perception = (model_perception == ground_truth_should_see).sum()
    total = len(ground_truth_should_see)

    return {
        "correct_perception_ratio": correct_perception / total if total > 0 else 0.0,
        "total_illusions": total,
        "correctly_perceived": correct_perception,
    }
