"""
Predictive Coding Neural Network Architecture.

Implements a bidirectional neural network with:
- Feedforward pathway: Hierarchical feature extraction via Conv layers
- Feedback pathway: Reconstruction via ConvTranspose layers
- Predictive coding: Iterative error minimization across layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import NETWORK_ARCHITECTURE, INPUT_SIZE_TO_FC
from checkpoint_utils import initialize_feature_tensors


class Net(nn.Module):
    """
    Predictive Coding Neural Network.

    Architecture:
    - 4 Conv layers with pooling: 3 -> 6 -> 16 -> 32 -> 128 channels
    - 3 FC layers for classification
    - Mirrored ConvTranspose layers for reconstruction
    """

    def __init__(self, num_classes: int, input_size: int = 32):
        """
        Initialize network.

        Args:
            num_classes: Number of output classes
            input_size: Input image size (32 or 128)
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        if input_size not in INPUT_SIZE_TO_FC:
            raise ValueError(f"Input size {input_size} not supported. Use 32 or 128.")

        self.fc_input_size = INPUT_SIZE_TO_FC[input_size]

        # Feedforward pathway (Conv layers)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # Fully connected layers (classification)
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Feedback pathway (Dense layer feedback)
        self.fc3_fb = nn.Linear(num_classes, 256)
        self.fc2_fb = nn.Linear(256, 1024)
        self.fc1_fb = nn.Linear(1024, self.fc_input_size)

        # Feedback pathway (ConvTranspose layers)
        self.deconv4_fb = nn.ConvTranspose2d(128, 32, kernel_size=5, stride=1, padding=2)
        self.deconv3_fb = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.deconv2_fb = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=1, padding=2)
        self.deconv1_fb = nn.ConvTranspose2d(6, 3, kernel_size=5, stride=1, padding=2)

        # Upsampling layers
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_nearest = nn.Upsample(scale_factor=2, mode="nearest")

    def feedforward_pass(self, x, ft_AB=None, ft_BC=None, ft_CD=None, ft_DE=None):
        """
        Feedforward pass with all layers (classification).

        Args:
            x: Input tensor (batch_size, 3, height, width)
            ft_AB, ft_BC, ft_CD, ft_DE: Feature tensors (unused, kept for API compatibility)

        Returns:
            Tuple of layer features and classification output
        """
        ft_AB = self.conv1(x)
        pooled_ft_AB, _ = self.pool(F.relu(ft_AB))

        ft_BC = self.conv2(pooled_ft_AB)
        pooled_ft_BC, _ = self.pool(F.relu(ft_BC))

        ft_CD = self.conv3(pooled_ft_BC)
        pooled_ft_CD, _ = self.pool(F.relu(ft_CD))

        ft_DE = self.conv4(pooled_ft_CD)
        pooled_ft_DE, _ = self.pool(F.relu(ft_DE))

        ft_DE_flat = torch.flatten(pooled_ft_DE, 1)
        ft_EF = self.fc1(ft_DE_flat)
        relu_EF = F.relu(ft_EF)

        ft_FG = self.fc2(relu_EF)
        relu_FG = F.relu(ft_FG)

        output = self.fc3(relu_FG)

        return ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG, output

    def feedforward_pass_no_dense(self, x, ft_AB=None, ft_BC=None, ft_CD=None, ft_DE=None):
        """
        Feedforward pass without dense layers (reconstruction only).

        Args:
            x: Input tensor
            ft_AB, ft_BC, ft_CD, ft_DE: Feature tensors (unused, kept for API compatibility)

        Returns:
            Tuple of convolutional layer features
        """
        ft_AB = self.conv1(x)
        pooled_ft_AB, _ = self.pool(F.relu(ft_AB))

        ft_BC = self.conv2(pooled_ft_AB)
        pooled_ft_BC, _ = self.pool(F.relu(ft_BC))

        ft_CD = self.conv3(pooled_ft_BC)
        pooled_ft_CD, _ = self.pool(F.relu(ft_CD))

        ft_DE = self.conv4(pooled_ft_CD)

        return ft_AB, ft_BC, ft_CD, ft_DE

    def feedback_pass(self, output, ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG):
        """
        Feedback pass for reconstruction.

        Args:
            output: Classification output
            ft_AB, ft_BC, ft_CD, ft_DE: Conv layer features
            ft_EF, ft_FG: Dense layer features

        Returns:
            Reconstructed image and intermediate features
        """
        ft_GF = self.fc3_fb(output)
        ft_FE = self.fc2_fb(ft_FG)
        ft_ED = self.fc1_fb(ft_EF)
        # Reshape based on input_size (spatial dims after 4 pooling layers = input_size // 16)
        spatial_dim = self.input_size // 16
        ft_ED = ft_ED.view(-1, 128, spatial_dim, spatial_dim)
        ft_ED = self.upsample_nearest(ft_ED)

        ft_DC = self.deconv4_fb(self.upsample(ft_DE))
        ft_CB = self.deconv3_fb(self.upsample(ft_CD))
        ft_BA = self.deconv2_fb(self.upsample(ft_BC))
        x = self.deconv1_fb(ft_AB)

        return ft_BA, ft_CB, ft_DC, ft_ED, ft_FE, ft_GF, x

    def _compute_scaling_factor(self, layer_dim: int, input_dim: int) -> float:
        """
        Compute dynamic scaling factor for error gradients.

        Based on: https://proceedings.neurips.cc/paper_files/paper/2021/file/...
        The scaling factor accounts for the ratio of receptive field to total neurons.

        Args:
            layer_dim: Dimension of current layer
            input_dim: Dimension of input

        Returns:
            Scaling factor
        """
        scaling = np.sqrt(np.square(input_dim) / layer_dim)
        return np.round(scaling)

    def predictive_coding_pass(
        self,
        x,
        ft_AB,
        ft_BC,
        ft_CD,
        ft_DE,
        ft_EF,
        beta,
        gamma,
        alpha,
        batch_size,
    ):
        """
        Predictive coding pass for classification.

        Implements iterative error minimization:
        ft_t = gamma*forward(x) + (1-gamma-beta)*ft_prev + beta*feedback(ft_next) - alpha*error_gradient

        Args:
            x: Input image
            ft_AB, ft_BC, ft_CD, ft_DE, ft_EF: Feature tensors
            beta: Backward contribution weights [[beta1, beta2, beta3, beta4], ...]
            gamma: Forward contribution weights [[gamma1, gamma2, gamma3, gamma4], ...]
            alpha: Error gradient weights [[alpha1, alpha2, alpha3, alpha4], ...]
            batch_size: Batch size

        Returns:
            Prediction, updated features, and loss
        """
        gamma_AB_fwd, gamma_BC_fwd, gamma_CD_fwd, gamma_DE_fwd = gamma[0]
        beta_AB_bck, beta_BC_bck, beta_CD_bck, beta_DE_bck = beta[0]
        alpha_AB, alpha_BC, alpha_CD, alpha_DE = alpha[0]

        _, _, h_in, w_in = x.shape

        # Layer AB: Input reconstruction error
        errorB = F.mse_loss(self.deconv1_fb(ft_AB), x)
        reconstructionB = torch.autograd.grad(errorB, ft_AB, retain_graph=True)[0]
        scalingB = self._compute_scaling_factor(h_in * w_in * 3,
                                                 self.conv1.kernel_size[0] * self.conv1.in_channels)

        ft_AB_pc = (
            gamma_AB_fwd * self.conv1(x)
            + (1 - gamma_AB_fwd - beta_AB_bck) * ft_AB
            + beta_AB_bck * self.deconv2_fb(self.upsample(ft_BC))
            - alpha_AB * scalingB * batch_size * reconstructionB
        )

        # Layer BC
        errorC = F.mse_loss(self.deconv2_fb(self.upsample(ft_BC)), ft_AB)
        reconstructionC = torch.autograd.grad(errorC, ft_BC, retain_graph=True)[0]
        pooled_ft_AB_pc, _ = self.pool(F.relu(ft_AB_pc))
        scalingC = self._compute_scaling_factor((h_in // 2) * (w_in // 2) * 6,
                                                 self.conv2.kernel_size[0] * self.conv2.in_channels)

        ft_BC_pc = (
            gamma_BC_fwd * self.conv2(pooled_ft_AB_pc)
            + (1 - gamma_BC_fwd - beta_BC_bck) * ft_BC
            + beta_BC_bck * self.deconv3_fb(self.upsample(ft_CD))
            - alpha_BC * scalingC * batch_size * reconstructionC
        )

        # Layer CD
        errorD = F.mse_loss(self.deconv3_fb(self.upsample(ft_CD)), ft_BC)
        reconstructionD = torch.autograd.grad(errorD, ft_CD, retain_graph=True)[0]
        pooled_ft_BC_pc, _ = self.pool(F.relu(ft_BC_pc))
        scalingD = self._compute_scaling_factor((h_in // 4) * (w_in // 4) * 16,
                                                 self.conv3.kernel_size[0] * self.conv3.in_channels)

        ft_CD_pc = (
            gamma_CD_fwd * self.conv3(pooled_ft_BC_pc)
            + (1 - gamma_CD_fwd - beta_CD_bck) * ft_CD
            + beta_CD_bck * self.deconv4_fb(self.upsample(ft_DE))
            - alpha_CD * scalingD * batch_size * reconstructionD
        )

        # Layer DE
        errorE = F.mse_loss(self.deconv4_fb(self.upsample(ft_DE)), ft_CD)
        reconstructionE = torch.autograd.grad(errorE, ft_DE, retain_graph=True)[0]
        pooled_ft_CD_pc, _ = self.pool(F.relu(ft_CD_pc))
        scalingE = self._compute_scaling_factor((h_in // 8) * (w_in // 8) * 32,
                                                 self.conv4.kernel_size[0] * self.conv4.in_channels)

        ft_DE_pc = (
            gamma_DE_fwd * self.conv4(pooled_ft_CD_pc)
            + (1 - gamma_DE_fwd) * ft_DE
            - alpha_DE * scalingE * batch_size * reconstructionE
        )

        # Classification via FC layers
        pooled_ft_DE, _ = self.pool(F.relu(ft_DE_pc))
        ft_DE_flat = torch.flatten(pooled_ft_DE, 1)
        ft_EF_pc = self.fc1(ft_DE_flat)
        relu_EF = F.relu(ft_EF_pc)
        ft_FG_pc = self.fc2(relu_EF)
        relu_FG = F.relu(ft_FG_pc)
        output = self.fc3(relu_FG)

        loss_of_layers = errorB + errorC + errorD + errorE

        return output, ft_AB_pc, ft_BC_pc, ft_CD_pc, ft_DE_pc, ft_EF_pc, loss_of_layers

    def recon_predictive_coding_pass(
        self,
        x,
        ft_AB,
        ft_BC,
        ft_CD,
        ft_DE,
        beta,
        gamma,
        alpha,
        batch_size,
    ):
        """
        Predictive coding pass for reconstruction (no classification).

        Args:
            x: Input image
            ft_AB, ft_BC, ft_CD, ft_DE: Feature tensors
            beta: Backward contribution weights
            gamma: Forward contribution weights
            alpha: Error gradient weights
            batch_size: Batch size

        Returns:
            Updated features and loss
        """
        gamma_AB_fwd, gamma_BC_fwd, gamma_CD_fwd, gamma_DE_fwd = gamma[0]
        beta_AB_bck, beta_BC_bck, beta_CD_bck, beta_DE_bck = beta[0]
        alpha_AB, alpha_BC, alpha_CD, alpha_DE = alpha[0]

        _, _, h_in, w_in = x.shape

        # Compute errors and gradients for each layer
        errorB = F.mse_loss(self.deconv1_fb(ft_AB), x)
        reconstructionB = torch.autograd.grad(errorB, ft_AB, retain_graph=True)[0]
        scalingB = self._compute_scaling_factor(h_in * w_in * 3,
                                                 self.conv1.kernel_size[0] * self.conv1.in_channels)

        ft_AB_pc = (
            gamma_AB_fwd * self.conv1(x)
            + (1 - gamma_AB_fwd - beta_AB_bck) * ft_AB
            + beta_AB_bck * self.deconv2_fb(self.upsample(ft_BC))
            - alpha_AB * scalingB * batch_size * reconstructionB
        )

        errorC = F.mse_loss(self.deconv2_fb(self.upsample(ft_BC)), ft_AB)
        reconstructionC = torch.autograd.grad(errorC, ft_BC, retain_graph=True)[0]
        pooled_ft_AB_pc, _ = self.pool(F.relu(ft_AB_pc))
        scalingC = self._compute_scaling_factor((h_in // 2) * (w_in // 2) * 6,
                                                 self.conv2.kernel_size[0] * self.conv2.in_channels)

        ft_BC_pc = (
            gamma_BC_fwd * self.conv2(pooled_ft_AB_pc)
            + (1 - gamma_BC_fwd - beta_BC_bck) * ft_BC
            + beta_BC_bck * self.deconv3_fb(self.upsample(ft_CD))
            - alpha_BC * scalingC * batch_size * reconstructionC
        )

        errorD = F.mse_loss(self.deconv3_fb(self.upsample(ft_CD)), ft_BC)
        reconstructionD = torch.autograd.grad(errorD, ft_CD, retain_graph=True)[0]
        pooled_ft_BC_pc, _ = self.pool(F.relu(ft_BC_pc))
        scalingD = self._compute_scaling_factor((h_in // 4) * (w_in // 4) * 16,
                                                 self.conv3.kernel_size[0] * self.conv3.in_channels)

        ft_CD_pc = (
            gamma_CD_fwd * self.conv3(pooled_ft_BC_pc)
            + (1 - gamma_CD_fwd - beta_CD_bck) * ft_CD
            + beta_CD_bck * self.deconv4_fb(self.upsample(ft_DE))
            - alpha_CD * scalingD * batch_size * reconstructionD
        )

        errorE = F.mse_loss(self.deconv4_fb(self.upsample(ft_DE)), ft_CD)
        reconstructionE = torch.autograd.grad(errorE, ft_DE, retain_graph=True)[0]
        pooled_ft_CD_pc, _ = self.pool(F.relu(ft_CD_pc))
        scalingE = self._compute_scaling_factor((h_in // 8) * (w_in // 8) * 32,
                                                 self.conv4.kernel_size[0] * self.conv4.in_channels)

        ft_DE_pc = (
            gamma_DE_fwd * self.conv4(pooled_ft_CD_pc)
            + (1 - gamma_DE_fwd) * ft_DE
            - alpha_DE * scalingE * batch_size * reconstructionE
        )

        loss_of_layers = errorB + errorC + errorD + errorE

        return ft_AB_pc, ft_BC_pc, ft_CD_pc, ft_DE_pc, loss_of_layers
