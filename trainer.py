"""
Unified training interface for different training modes.
Consolidates training logic from illusion_pc_train.py and recon_pc_train.py.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Callable
from torch.utils.data import DataLoader
from network import Net
from checkpoint_utils import (
    initialize_feature_tensors,
    save_model_checkpoint,
    get_layer_parameters,
)
from metrics import compute_classification_accuracy, compute_reconstruction_loss
from constants import NOISE_LEVELS, DEFAULT_TIMESTEPS
from add_noise import noisy_img


class PredictiveCodingTrainer:
    """Unified trainer for predictive coding networks."""

    def __init__(
        self,
        net: Net,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        training_condition: str = "illusion_pc_train",
    ):
        """
        Initialize trainer.

        Args:
            net: Network to train
            device: Device to train on
            optimizer: Optimizer
            training_condition: Type of training ("illusion_pc_train", "recon_pc_train")
        """
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.training_condition = training_condition

        if training_condition not in ["illusion_pc_train", "recon_pc_train"]:
            raise ValueError(f"Unknown training condition: {training_condition}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        timesteps: int = DEFAULT_TIMESTEPS,
        gammaset: list = None,
        betaset: list = None,
        alphaset: list = None,
        noise_levels: list = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            timesteps: Number of predictive coding timesteps
            gammaset: Forward contribution weights
            betaset: Backward contribution weights
            alphaset: Error gradient weights
            noise_levels: List of noise levels to try

        Returns:
            Dictionary with training metrics
        """
        if gammaset is None:
            gammaset = [[0.33, 0.33, 0.33, 0.33]]
        if betaset is None:
            betaset = [[0.33, 0.33, 0.33, 0.33]]
        if alphaset is None:
            alphaset = [[0.01, 0.01, 0.01, 0.01]]
        if noise_levels is None:
            noise_levels = NOISE_LEVELS

        self.net.train()
        total_loss = 0.0
        total_accuracy = 0.0
        batch_count = 0

        for batch_idx, batch_data in enumerate(dataloader):
            if self.training_condition == "illusion_pc_train":
                loss, accuracy = self._train_batch_classification(
                    batch_data,
                    timesteps,
                    gammaset,
                    betaset,
                    alphaset,
                    noise_levels,
                )
            else:  # recon_pc_train
                loss = self._train_batch_reconstruction(
                    batch_data,
                    timesteps,
                    gammaset,
                    betaset,
                    alphaset,
                    noise_levels,
                )
                accuracy = 0.0

            total_loss += loss
            total_accuracy += accuracy
            batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return {
            "loss": total_loss / batch_count,
            "accuracy": total_accuracy / batch_count,
        }

    def _train_batch_classification(
        self,
        batch_data: Tuple,
        timesteps: int,
        gammaset: list,
        betaset: list,
        alphaset: list,
        noise_levels: list,
    ) -> Tuple[float, float]:
        """Train a single batch for classification."""
        images, labels, class_names, should_sees = batch_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.size(0)

        # Initialize feature tensors
        ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG = initialize_feature_tensors(
            batch_size,
            images.size(2),
            images.size(3),
            self.device,
            include_dense=True,
        )

        total_loss = 0.0
        all_predictions = []

        for noise_level in noise_levels:
            # Add noise
            if noise_level > 0:
                noisy_images = noisy_img(images, "gauss", noise_level)
            else:
                noisy_images = images

            ft_AB_temp = torch.zeros_like(ft_AB)
            ft_BC_temp = torch.zeros_like(ft_BC)
            ft_CD_temp = torch.zeros_like(ft_CD)
            ft_DE_temp = torch.zeros_like(ft_DE)
            ft_EF_temp = torch.zeros_like(ft_EF)
            ft_FG_temp = torch.zeros_like(ft_FG)

            # Enable gradients for predictive coding error computation
            ft_AB_temp.requires_grad_(True)
            ft_BC_temp.requires_grad_(True)
            ft_CD_temp.requires_grad_(True)
            ft_DE_temp.requires_grad_(True)

            # Predictive coding iterations
            for t in range(timesteps):
                self.optimizer.zero_grad()

                output, ft_AB_temp, ft_BC_temp, ft_CD_temp, ft_DE_temp, ft_EF_temp, loss = (
                    self.net.predictive_coding_pass(
                        noisy_images,
                        ft_AB_temp,
                        ft_BC_temp,
                        ft_CD_temp,
                        ft_DE_temp,
                        ft_EF_temp,
                        betaset,
                        gammaset,
                        alphaset,
                        batch_size,
                    )
                )

                # Classification loss
                classification_loss = torch.nn.functional.cross_entropy(output, labels)
                total_loss_t = loss + classification_loss

                total_loss_t.backward()
                self.optimizer.step()
                total_loss += total_loss_t.item()

            all_predictions.append(output.detach())

        accuracy = compute_classification_accuracy(all_predictions[-1], labels)
        return total_loss / len(noise_levels), accuracy

    def _train_batch_reconstruction(
        self,
        batch_data: Tuple,
        timesteps: int,
        gammaset: list,
        betaset: list,
        alphaset: list,
        noise_levels: list,
    ) -> float:
        """Train a single batch for reconstruction."""
        images, _, _, _ = batch_data
        images = images.to(self.device)
        batch_size = images.size(0)

        # Initialize feature tensors (no dense layers for reconstruction)
        ft_AB, ft_BC, ft_CD, ft_DE = initialize_feature_tensors(
            batch_size,
            images.size(2),
            images.size(3),
            self.device,
            include_dense=False,
        )

        total_loss = 0.0

        for noise_level in noise_levels:
            if noise_level > 0:
                noisy_images = noisy_img(images, noise_level, "gaussian")
            else:
                noisy_images = images

            ft_AB_temp = torch.zeros_like(ft_AB)
            ft_BC_temp = torch.zeros_like(ft_BC)
            ft_CD_temp = torch.zeros_like(ft_CD)
            ft_DE_temp = torch.zeros_like(ft_DE)

            # Enable gradients for predictive coding error computation
            ft_AB_temp.requires_grad_(True)
            ft_BC_temp.requires_grad_(True)
            ft_CD_temp.requires_grad_(True)
            ft_DE_temp.requires_grad_(True)

            for t in range(timesteps):
                self.optimizer.zero_grad()

                ft_AB_temp, ft_BC_temp, ft_CD_temp, ft_DE_temp, pc_loss = (
                    self.net.recon_predictive_coding_pass(
                        noisy_images,
                        ft_AB_temp,
                        ft_BC_temp,
                        ft_CD_temp,
                        ft_DE_temp,
                        betaset,
                        gammaset,
                        alphaset,
                        batch_size,
                    )
                )

                pc_loss.backward()
                self.optimizer.step()
                total_loss += pc_loss.item()

        return total_loss / len(noise_levels)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        timesteps: int = DEFAULT_TIMESTEPS,
        gammaset: list = None,
        betaset: list = None,
        alphaset: list = None,
    ) -> Dict[str, float]:
        """
        Evaluate on a dataset.

        Args:
            dataloader: Evaluation dataloader
            timesteps: Number of timesteps
            gammaset, betaset, alphaset: Pattern parameters

        Returns:
            Dictionary with evaluation metrics
        """
        if gammaset is None:
            gammaset = [[0.33, 0.33, 0.33, 0.33]]
        if betaset is None:
            betaset = [[0.33, 0.33, 0.33, 0.33]]
        if alphaset is None:
            alphaset = [[0.01, 0.01, 0.01, 0.01]]

        self.net.eval()
        total_accuracy = 0.0
        total_loss = 0.0
        batch_count = 0

        for batch_data in dataloader:
            images = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            batch_size = images.size(0)

            ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG = initialize_feature_tensors(
                batch_size,
                images.size(2),
                images.size(3),
                self.device,
                include_dense=True,
            )

            # Enable gradients for predictive coding error computation
            ft_AB.requires_grad_(True)
            ft_BC.requires_grad_(True)
            ft_CD.requires_grad_(True)
            ft_DE.requires_grad_(True)

            # Single predictive coding pass (no noise for evaluation)
            output, _, _, _, _, _, loss = self.net.predictive_coding_pass(
                images,
                ft_AB,
                ft_BC,
                ft_CD,
                ft_DE,
                ft_EF,
                betaset,
                gammaset,
                alphaset,
                batch_size,
            )

            classification_loss = torch.nn.functional.cross_entropy(output, labels)
            accuracy = compute_classification_accuracy(output, labels)

            total_accuracy += accuracy
            total_loss += classification_loss.item()
            batch_count += 1

        return {
            "eval_accuracy": total_accuracy / batch_count,
            "eval_loss": total_loss / batch_count,
        }
