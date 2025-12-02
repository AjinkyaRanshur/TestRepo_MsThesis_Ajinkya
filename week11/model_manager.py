
"""
Model Manager - Centralized model loading, saving, and tracking
"""
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from network import Net

class ModelManager:
    """Manages model checkpoints, metadata, and tracking."""


     def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.recon_dir = self.base_dir / "recon_models"
        self.class_dir = self.base_dir / "classification_models"
        self.metadata_file = self.base_dir / "model_registry.json"
        
        # Create directories
        for d in [self.recon_dir, self.class_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize registry
        self.registry = self._load_registry()


     def _load_registry(self) -> Dict:
        """Load model registry from JSON."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"reconstruction": {}, "classification": {}}



































