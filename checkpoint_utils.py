"""
Checkpoint utilities for saving and loading model states.
Centralizes all checkpoint operations to reduce duplication.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional, List


def initialize_feature_tensors(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    include_dense: bool = True,
) -> tuple:
    """
    Initialize feature tensors for predictive coding.

    Args:
        batch_size: Batch size
        height: Image height
        width: Image width
        device: torch device
        include_dense: If True, also initialize FC layer feature tensors

    Returns:
        Tuple of (ft_AB, ft_BC, ft_CD, ft_DE) or
        (ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG) if include_dense=True
    """
    # Convolutional layer feature tensors
    ft_AB = torch.zeros(batch_size, 6, height, width, device=device)
    ft_BC = torch.zeros(batch_size, 16, height // 2, width // 2, device=device)
    ft_CD = torch.zeros(batch_size, 32, height // 4, width // 4, device=device)
    ft_DE = torch.zeros(batch_size, 128, height // 8, width // 8, device=device)

    if include_dense:
        # Calculate FC input size (128 * 8 * 8 for 128x128, or 128 * 2 * 2 for 32x32)
        fc_input_size = 128 * (height // 64) * (width // 64)
        ft_EF = torch.zeros(batch_size, 1024, device=device)
        ft_FG = torch.zeros(batch_size, 256, device=device)
        return ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG

    return ft_AB, ft_BC, ft_CD, ft_DE


def save_model_checkpoint(
    net: torch.nn.Module,
    checkpoint_path: str,
    include_dense: bool = True,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint for either reconstruction or classification training.

    Args:
        net: Neural network module
        checkpoint_path: Path to save checkpoint
        include_dense: If True, save FC layers; if False, save only conv layers
        metadata: Optional metadata to include (e.g., epoch, loss)
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "conv1": net.conv1.state_dict(),
        "conv2": net.conv2.state_dict(),
        "conv3": net.conv3.state_dict(),
        "conv4": net.conv4.state_dict(),
        "deconv1_fb": net.deconv1_fb.state_dict(),
        "deconv2_fb": net.deconv2_fb.state_dict(),
        "deconv3_fb": net.deconv3_fb.state_dict(),
        "deconv4_fb": net.deconv4_fb.state_dict(),
    }

    if include_dense:
        checkpoint.update({
            "fc1": net.fc1.state_dict(),
            "fc2": net.fc2.state_dict(),
            "fc3": net.fc3.state_dict(),
            "fc1_fb": net.fc1_fb.state_dict(),
            "fc2_fb": net.fc2_fb.state_dict(),
            "fc3_fb": net.fc3_fb.state_dict(),
        })

    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, checkpoint_path)


def load_model_checkpoint(
    net: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    include_dense: bool = True,
) -> None:
    """
    Load model checkpoint.

    Args:
        net: Neural network module to load into
        checkpoint_path: Path to checkpoint file
        device: Device to map to
        include_dense: If True, load FC layers; if False, load only conv layers
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load convolutional layers
    net.conv1.load_state_dict(checkpoint["conv1"])
    net.conv2.load_state_dict(checkpoint["conv2"])
    net.conv3.load_state_dict(checkpoint["conv3"])
    net.conv4.load_state_dict(checkpoint["conv4"])
    net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
    net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
    net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
    net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])

    if include_dense and "fc1" in checkpoint:
        net.fc1.load_state_dict(checkpoint["fc1"])
        net.fc2.load_state_dict(checkpoint["fc2"])
        net.fc3.load_state_dict(checkpoint["fc3"])
        net.fc1_fb.load_state_dict(checkpoint["fc1_fb"])
        net.fc2_fb.load_state_dict(checkpoint["fc2_fb"])
        net.fc3_fb.load_state_dict(checkpoint["fc3_fb"])


def load_partial_checkpoint(
    net: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    freeze_conv_layers: bool = False,
) -> None:
    """
    Load checkpoint and optionally freeze convolutional layers.
    Used for transfer learning scenarios.

    Args:
        net: Neural network module
        checkpoint_path: Path to checkpoint
        device: Device to map to
        freeze_conv_layers: If True, freeze conv layers after loading
    """
    load_model_checkpoint(net, checkpoint_path, device, include_dense=True)

    if freeze_conv_layers:
        for param in net.conv1.parameters():
            param.requires_grad = False
        for param in net.conv2.parameters():
            param.requires_grad = False
        for param in net.conv3.parameters():
            param.requires_grad = False
        for param in net.conv4.parameters():
            param.requires_grad = False


def get_layer_parameters(
    net: torch.nn.Module,
    layer_names: Optional[List[str]] = None,
) -> List[torch.nn.parameter.Parameter]:
    """
    Get parameters for specific layers.

    Args:
        net: Neural network module
        layer_names: List of layer names to get parameters from.
                    If None, returns all parameters.
                    Use "fc" to get only FC layer parameters.

    Returns:
        List of parameters
    """
    if layer_names is None:
        return list(net.parameters())

    params = []
    if "fc" in layer_names:
        params.extend(net.fc1.parameters())
        params.extend(net.fc2.parameters())
        params.extend(net.fc3.parameters())
        params.extend(net.fc1_fb.parameters())
        params.extend(net.fc2_fb.parameters())
        params.extend(net.fc3_fb.parameters())

    if "conv" in layer_names:
        params.extend(net.conv1.parameters())
        params.extend(net.conv2.parameters())
        params.extend(net.conv3.parameters())
        params.extend(net.conv4.parameters())

    if "deconv" in layer_names:
        params.extend(net.deconv1_fb.parameters())
        params.extend(net.deconv2_fb.parameters())
        params.extend(net.deconv3_fb.parameters())
        params.extend(net.deconv4_fb.parameters())

    return params
