"""
Unified test runner for trajectory, pattern, and grid search testing.
Consolidates testing logic from test_workflow.py, pattern_testing.py, grid_search_testing.py.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from network_refactored import Net
from checkpoint_utils import initialize_feature_tensors, load_model_checkpoint
from metrics import compute_classification_accuracy, aggregate_accuracies_across_seeds
from constants import DEFAULT_TIMESTEPS
from pattern_manager import PatternManager


class TestRunner:
    """Unified test runner for different testing scenarios."""

    def __init__(
        self,
        net: Net,
        device: torch.device,
    ):
        """
        Initialize test runner.

        Args:
            net: Network to test
            device: Device to test on
        """
        self.net = net
        self.device = device
        self.pattern_manager = PatternManager()

    @torch.no_grad()
    def run_trajectory_test(
        self,
        dataloader: DataLoader,
        timesteps: int = DEFAULT_TIMESTEPS,
        patterns: Optional[Dict] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run trajectory testing (accuracy across timesteps).

        Args:
            dataloader: Test dataloader
            timesteps: Maximum number of timesteps to test
            patterns: Pattern configuration (gamma, beta, alpha)

        Returns:
            Dictionary mapping timestep to accuracy
        """
        if patterns is None:
            patterns = {"gamma": [0.33, 0.33, 0.33, 0.33],
                       "beta": [0.33, 0.33, 0.33, 0.33],
                       "alpha": [0.01, 0.01, 0.01, 0.01]}

        gamma = [patterns["gamma"]]
        beta = [patterns["beta"]]
        alpha = [patterns["alpha"]]

        self.net.eval()
        timestep_accuracies = {t: [] for t in range(1, timesteps + 1)}

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

            ft_AB_temp = ft_AB.clone()
            ft_BC_temp = ft_BC.clone()
            ft_CD_temp = ft_CD.clone()
            ft_DE_temp = ft_DE.clone()
            ft_EF_temp = ft_EF.clone()
            ft_FG_temp = ft_FG.clone()

            for t in range(1, timesteps + 1):
                output, ft_AB_temp, ft_BC_temp, ft_CD_temp, ft_DE_temp, ft_EF_temp, _ = (
                    self.net.predictive_coding_pass(
                        images,
                        ft_AB_temp,
                        ft_BC_temp,
                        ft_CD_temp,
                        ft_DE_temp,
                        ft_EF_temp,
                        beta,
                        gamma,
                        alpha,
                        batch_size,
                    )
                )

                accuracy = compute_classification_accuracy(output, labels)
                timestep_accuracies[t].append(accuracy)

        # Average across batches
        return {t: np.mean(acc) for t, acc in timestep_accuracies.items()}

    @torch.no_grad()
    def run_pattern_test(
        self,
        dataloader: DataLoader,
        pattern_names: Optional[List[str]] = None,
        timesteps: int = DEFAULT_TIMESTEPS,
    ) -> Dict[str, float]:
        """
        Run pattern testing (accuracy across different patterns).

        Args:
            dataloader: Test dataloader
            pattern_names: List of pattern names to test
            timesteps: Number of timesteps for each pattern

        Returns:
            Dictionary mapping pattern names to accuracies
        """
        if pattern_names is None:
            pattern_names = self.pattern_manager.get_available_patterns()

        self.net.eval()
        pattern_accuracies = {}

        for pattern_name in pattern_names:
            pattern = self.pattern_manager.get_pattern(pattern_name)

            self.net.eval()
            batch_accuracies = []

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

                ft_AB_temp = ft_AB.clone()
                ft_BC_temp = ft_BC.clone()
                ft_CD_temp = ft_CD.clone()
                ft_DE_temp = ft_DE.clone()
                ft_EF_temp = ft_EF.clone()
                ft_FG_temp = ft_FG.clone()

                for t in range(timesteps):
                    output, ft_AB_temp, ft_BC_temp, ft_CD_temp, ft_DE_temp, ft_EF_temp, _ = (
                        self.net.predictive_coding_pass(
                            images,
                            ft_AB_temp,
                            ft_BC_temp,
                            ft_CD_temp,
                            ft_DE_temp,
                            ft_EF_temp,
                            [pattern["beta"]],
                            [pattern["gamma"]],
                            [pattern["alpha"]],
                            batch_size,
                        )
                    )

                accuracy = compute_classification_accuracy(output, labels)
                batch_accuracies.append(accuracy)

            pattern_accuracies[pattern_name] = np.mean(batch_accuracies)

        return pattern_accuracies

    @torch.no_grad()
    def run_grid_search_test(
        self,
        dataloader: DataLoader,
        gamma_range: np.ndarray,
        beta_range: np.ndarray,
        timesteps: int = DEFAULT_TIMESTEPS,
    ) -> Dict[Tuple[float, float], float]:
        """
        Run grid search test over gamma and beta parameters.

        Args:
            dataloader: Test dataloader
            gamma_range: Range of gamma values to test
            beta_range: Range of beta values to test
            timesteps: Number of timesteps for each configuration

        Returns:
            Dictionary mapping (gamma, beta) tuples to accuracies
        """
        self.net.eval()
        grid_results = {}

        for gamma in gamma_range:
            for beta in beta_range:
                gamma_pattern = [gamma, gamma, gamma, gamma]
                beta_pattern = [beta, beta, beta, beta]
                alpha_pattern = [0.01, 0.01, 0.01, 0.01]

                batch_accuracies = []

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

                    ft_AB_temp = ft_AB.clone()
                    ft_BC_temp = ft_BC.clone()
                    ft_CD_temp = ft_CD.clone()
                    ft_DE_temp = ft_DE.clone()
                    ft_EF_temp = ft_EF.clone()
                    ft_FG_temp = ft_FG.clone()

                    for t in range(timesteps):
                        output, ft_AB_temp, ft_BC_temp, ft_CD_temp, ft_DE_temp, ft_EF_temp, _ = (
                            self.net.predictive_coding_pass(
                                images,
                                ft_AB_temp,
                                ft_BC_temp,
                                ft_CD_temp,
                                ft_DE_temp,
                                ft_EF_temp,
                                [beta_pattern],
                                [gamma_pattern],
                                [alpha_pattern],
                                batch_size,
                            )
                        )

                    accuracy = compute_classification_accuracy(output, labels)
                    batch_accuracies.append(accuracy)

                grid_results[(gamma, beta)] = np.mean(batch_accuracies)

        return grid_results
