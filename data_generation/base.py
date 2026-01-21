"""
Abstract base class for illusion dataset generators.
Provides common interface and utilities.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import os
import csv
from PIL import Image


class IllusionDatasetGenerator(ABC):
    """
    Abstract base class for generating visual illusion datasets.

    Subclasses should implement dataset-specific generation logic
    while inheriting common utilities for file management and metadata.
    """

    def __init__(self, output_dir: str, img_size: int = 128):
        """
        Initialize generator.

        Args:
            output_dir: Directory to save generated images
            img_size: Canvas size (width = height)
        """
        self.output_dir = output_dir
        self.img_size = img_size
        self.metadata = []

    def create_output_directory(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)

    def save_image(self, img: Image.Image, filename: str, subdirectory: str = None):
        """
        Save image to output directory.

        Args:
            img: PIL Image to save
            filename: Filename (without path)
            subdirectory: Optional subdirectory within output_dir
        """
        if subdirectory:
            save_dir = os.path.join(self.output_dir, subdirectory)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir

        filepath = os.path.join(save_dir, filename)
        img.save(filepath)
        return filepath

    def add_metadata(self, entry: Dict):
        """
        Add entry to metadata list.

        Args:
            entry: Dictionary with metadata fields
        """
        self.metadata.append(entry)

    def save_metadata(self, filename: str = "dataset_metadata.csv"):
        """
        Save metadata to CSV file.

        Args:
            filename: CSV filename
        """
        if not self.metadata:
            print("No metadata to save")
            return

        csv_path = os.path.join(self.output_dir, filename)
        fieldnames = self.metadata[0].keys()

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metadata)

        print(f"Metadata saved to {csv_path}")

    @abstractmethod
    def generate_dataset(self) -> Tuple[int, int]:
        """
        Generate the complete dataset.

        Returns:
            Tuple of (total_images_generated, num_classes)
        """
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """
        Get list of class names for this dataset.

        Returns:
            List of class name strings
        """
        pass

    def print_summary(self, total_images: int):
        """
        Print generation summary.

        Args:
            total_images: Total number of images generated
        """
        print(f"\n{'='*60}")
        print(f"Dataset Generation Complete")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total images: {total_images}")
        print(f"Classes: {len(self.get_class_names())}")
        print(f"Image size: {self.img_size}x{self.img_size}")
        print(f"{'='*60}\n")
