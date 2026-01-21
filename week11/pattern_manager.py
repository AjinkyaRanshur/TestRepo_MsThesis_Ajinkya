"""
Pattern management system for predictive coding experiments.
Easily add new patterns without modifying core code.
"""

from typing import Dict, List, Tuple, Optional
from constants import TESTING_PATTERNS, DEFAULT_GAMMA_SET, DEFAULT_BETA_SET, DEFAULT_ALPHA_SET


class PatternManager:
    """Manages patterns for testing and training."""

    def __init__(self):
        """Initialize with default patterns."""
        self.patterns = TESTING_PATTERNS.copy()
        self.custom_patterns = {}

    def add_pattern(
        self,
        name: str,
        gamma: List[float],
        beta: List[float],
        alpha: Optional[List[float]] = None,
    ) -> None:
        """
        Add a custom pattern.

        Args:
            name: Pattern name
            gamma: Forward contribution weights [layer1, layer2, layer3, layer4]
            beta: Backward contribution weights [layer1, layer2, layer3, layer4]
            alpha: Error gradient weights (optional)
        """
        if len(gamma) != 4 or len(beta) != 4:
            raise ValueError(
                "Gamma and beta must have 4 values (one for each layer)"
            )

        self.custom_patterns[name] = {
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha or [0.01, 0.01, 0.01, 0.01],
        }

    def get_pattern(self, name: str) -> Dict[str, List[float]]:
        """
        Get pattern by name.

        Args:
            name: Pattern name

        Returns:
            Pattern dictionary with gamma, beta, alpha
        """
        if name in self.patterns:
            pattern = self.patterns[name]
            return {
                "gamma": pattern["gamma"],
                "beta": pattern["beta"],
                "alpha": [0.01, 0.01, 0.01, 0.01],  # Default alpha
            }

        if name in self.custom_patterns:
            return self.custom_patterns[name]

        raise ValueError(f"Pattern '{name}' not found")

    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern names."""
        return list(self.patterns.keys()) + list(self.custom_patterns.keys())

    def get_all_patterns(self) -> Dict[str, Dict[str, List[float]]]:
        """Get all patterns (built-in and custom)."""
        all_patterns = {}
        for name in self.patterns:
            all_patterns[name] = self.get_pattern(name)
        for name in self.custom_patterns:
            all_patterns[name] = self.custom_patterns[name]
        return all_patterns

    @staticmethod
    def create_custom_pattern_set(
        pattern_values: List[Tuple[float, float]],
        pattern_type: str = "gamma_sweep",
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Create a set of patterns for grid/sweep experiments.

        Args:
            pattern_values: List of (gamma, beta) tuples
            pattern_type: Type of sweep ("gamma_sweep", "beta_sweep", "grid", etc.)

        Returns:
            Dictionary of patterns
        """
        patterns = {}
        for i, (gamma, beta) in enumerate(pattern_values):
            name = f"{pattern_type}_{i}"
            patterns[name] = {
                "gamma": [gamma, gamma, gamma, gamma],
                "beta": [beta, beta, beta, beta],
                "alpha": [0.01, 0.01, 0.01, 0.01],
            }
        return patterns

    def export_patterns_to_config(self, patterns_dict: Dict) -> str:
        """
        Export patterns to configuration format (gammaset, betaset, alphaset).

        Args:
            patterns_dict: Dictionary of patterns

        Returns:
            Configuration string
        """
        gammaset = []
        betaset = []
        alphaset = []

        for name, pattern in patterns_dict.items():
            gammaset.append(pattern["gamma"])
            betaset.append(pattern["beta"])
            alphaset.append(pattern.get("alpha", [0.01, 0.01, 0.01, 0.01]))

        config_str = (
            f"gammaset = {gammaset}\n"
            f"betaset = {betaset}\n"
            f"alphaset = {alphaset}\n"
        )
        return config_str
