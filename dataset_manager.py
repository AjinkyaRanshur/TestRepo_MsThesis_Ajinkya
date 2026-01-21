"""
Dataset management system for loading and handling different datasets.
Easily add new datasets without modifying core code.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from constants import ILLUSION_DATASET_CLASSES, ALL_ILLUSION_CLASSES
from customdataset import SquareDataset


class DatasetManager:
    """Manages different datasets and their configurations."""

    def __init__(self):
        """Initialize with registered datasets."""
        self.datasets = {}
        self._register_default_datasets()

    def _register_default_datasets(self) -> None:
        """Register built-in datasets."""
        self.register_dataset(
            "illusion",
            {
                "type": "SquareDataset",
                "classes": ALL_ILLUSION_CLASSES,
                "description": "Visual illusion dataset with shapes and illusions",
            },
        )

    def register_dataset(
        self,
        name: str,
        config: Dict,
    ) -> None:
        """
        Register a new dataset.

        Args:
            name: Dataset name
            config: Dataset configuration dictionary with:
                - type: Dataset class name (e.g., "SquareDataset")
                - classes: List of class names
                - description: Human-readable description
        """
        self.datasets[name] = config

    def get_dataset_config(self, name: str) -> Dict:
        """Get configuration for a dataset."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        return self.datasets[name]

    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.datasets.keys())

    def load_dataset(
        self,
        name: str,
        csv_path: str,
        img_dir: str,
        transform: Optional[transforms.Compose] = None,
        subset_classes: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Load a dataset.

        Args:
            name: Dataset name
            csv_path: Path to CSV metadata file
            img_dir: Path to images directory
            transform: Optional image transforms
            subset_classes: Optional subset of classes to use

        Returns:
            Loaded dataset
        """
        config = self.get_dataset_config(name)
        dataset_type = config["type"]
        classes = subset_classes or config["classes"]

        if dataset_type == "SquareDataset":
            return SquareDataset(
                csv_file=csv_path,
                img_dir=img_dir,
                classes_for_use=classes,
                transform=transform,
            )

        raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def get_standard_transforms(
        image_size: int = 128,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> transforms.Compose:
        """
        Get standard image transforms.

        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std

        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def get_class_splits() -> Dict[str, List[str]]:
        """Get default class splits for illusion dataset."""
        return ILLUSION_DATASET_CLASSES

    @staticmethod
    def get_illusion_classes() -> List[str]:
        """Get illusion-specific classes."""
        return ILLUSION_DATASET_CLASSES["illusions"]

    @staticmethod
    def get_basic_shape_classes() -> List[str]:
        """Get basic shape classes."""
        return ILLUSION_DATASET_CLASSES["basic_shapes"]

    @staticmethod
    def is_illusion_class(class_name: str) -> bool:
        """Check if a class is an illusion class."""
        return class_name in ILLUSION_DATASET_CLASSES["illusions"]

    @staticmethod
    def get_perceived_class(
        class_name: str,
        should_see: bool,
    ) -> str:
        """
        Get the perceived class for an illusion.
        For illusion classes, returns the should_see value as the class.
        For other classes, returns the class name.

        Args:
            class_name: Original class name
            should_see: Whether the illusion should be perceived

        Returns:
            Perceived class name
        """
        if class_name in ILLUSION_DATASET_CLASSES["illusions"]:
            return "perceived" if should_see else "not_perceived"
        return class_name

    @staticmethod
    def create_class_subset(
        all_classes: List[str],
        exclude_classes: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Create a subset of classes.

        Args:
            all_classes: All available classes
            exclude_classes: Classes to exclude

        Returns:
            Subset of classes
        """
        if exclude_classes is None:
            return all_classes
        return [c for c in all_classes if c not in exclude_classes]
