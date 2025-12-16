"""
Model Manager - Centralized model tracking system
Tracks all models using their names as keys
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


class ModelTracker:
    def __init__(self, tracking_file="model_registry.json"):
        self.tracking_file = tracking_file
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry or create new one"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "metadata": {"last_updated": None}}
    
    def _save_registry(self):
        """Save registry to disk"""
        self.registry["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.tracking_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name: str, config: Dict):
        """Register a new model with its configuration"""
        self.registry["models"][model_name] = {
            "config": config,
            "status": "registered",
            "created_at": datetime.now().isoformat(),
            "training_started": None,
            "training_completed": None,
            "metrics": {},
            "checkpoint_path": None,
            "type": config.get("train_cond", "unknown")
        }
        
        self._save_registry()
        return model_name
    
    def update_status(self, model_name: str, status: str):
        """Update model training status"""
        if model_name in self.registry["models"]:
            self.registry["models"][model_name]["status"] = status
            
            if status == "training":
                self.registry["models"][model_name]["training_started"] = datetime.now().isoformat()
            elif status == "completed":
                self.registry["models"][model_name]["training_completed"] = datetime.now().isoformat()
            
            self._save_registry()
    
    def update_metrics(self, model_name: str, metrics: Dict):
        """Update model metrics"""
        if model_name in self.registry["models"]:
            self.registry["models"][model_name]["metrics"].update(metrics)
            self._save_registry()
 
    def set_checkpoint_path(self, model_name: str, path: str):
        """Set model checkpoint path"""
        if model_name in self.registry["models"]:
            self.registry["models"][model_name]["checkpoint_path"] = path
            self._save_registry()
    
    def get_model(self, model_name: str) -> Optional[Dict]:
        """Get model info by name"""
        return self.registry["models"].get(model_name)
    
    def get_models_by_type(self, model_type: str) -> List[Dict]:
        """Get all models of a specific type (e.g., 'recon_pc_train')"""
        return [
            {"name": name, **mdata}
            for name, mdata in self.registry["models"].items()
            if mdata.get("type") == model_type
        ]
    
    def get_completed_recon_models(self) -> List[Dict]:
        """Get all completed reconstruction models for classification training"""
        return [
            {"name": name, **mdata}
            for name, mdata in self.registry["models"].items()
            if mdata.get("type") == "recon_pc_train" and mdata.get("status") == "completed"
        ]
    def get_completed_classification_models(self) -> List[Dict]:
        """Get all completed classification models for testing"""
        return [
            {"name": name, **mdata}
            for name, mdata in self.registry["models"].items()
            if mdata.get("type") == "classification_training_shapes" and mdata.get("status") == "completed"
        ]

    def list_all_models(self, filter_status: Optional[str] = None) -> List[Dict]:
        """List all models, optionally filtered by status"""
        models = [
            {"name": name, **mdata}
            for name, mdata in self.registry["models"].items()
        ]
        
        if filter_status:
            models = [m for m in models if m.get("status") == filter_status]
        
        return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def print_model_summary(self, model_name: str):
        """Print a formatted summary of a model"""
        model = self.get_model(model_name)
        if not model:
            print(f"Model {model_name} not found")
            return
        
        print("\n" + "="*60)
        print(f"Model Name: {model_name}")
        print(f"Type: {model['type']}")
        print(f"Status: {model['status']}")
        print(f"Created: {model['created_at']}")
        
        if model.get('checkpoint_path'):
            print(f"Checkpoint: {model['checkpoint_path']}")
        
        if model.get('metrics'):
            print("\nMetrics:")
            for key, val in model['metrics'].items():
                print(f"  {key}: {val}")
        
        print("="*60 + "\n")
    
    def print_recon_models_table(self):
        """Print a table of all completed reconstruction models"""
        models = self.get_completed_recon_models()
        
        if not models:
            print("No completed reconstruction models found.")
            return
        
        print("\n" + "="*80)
        print("COMPLETED RECONSTRUCTION MODELS")
        print("="*80)
        print(f"{'Model Name':<50} {'Seed':<8} {'Status':<10}")
        print("-"*80)
        
        for model in models:
            name = model['name'][:48]
            config = model.get('config', {})
            seed = str(config.get('seed', 'N/A'))
            status = model.get('status', 'unknown')
            
            print(f"{name:<50} {seed:<8} {status:<10}")
        
        print("="*80 + "\n")
    
    def export_summary(self, output_file: str = "model_summary.txt"):
        """Export a text summary of all models"""
        with open(output_file, 'w') as f:
            f.write("MODEL TRACKING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, model_data in self.registry["models"].items():
                f.write(f"Model Name: {model_name}\n")
                f.write(f"  Type: {model_data['type']}\n")
                f.write(f"  Status: {model_data['status']}\n")
                f.write(f"  Created: {model_data['created_at']}\n")
                
                if model_data.get('checkpoint_path'):
                    f.write(f"  Checkpoint: {model_data['checkpoint_path']}\n")
                
                f.write("\n")
        
        print(f"Summary exported to {output_file}")


# Convenience functions for quick access
_tracker_instance = None

def get_tracker() -> ModelTracker:
    """Get global tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ModelTracker()
    return _tracker_instance


def register_model(model_name: str, config: Dict):
    """Quick register function"""
    return get_tracker().register_model(model_name, config)


def update_model_status(model_name: str, status: str):
    """Quick status update function"""
    get_tracker().update_status(model_name, status)


def get_completed_recon_models() -> List[Dict]:
    """Quick access to completed reconstruction models"""
    return get_tracker().get_completed_recon_models()


def get_completed_classification_models() -> List[Dict]:
    """Quick access to completed reconstruction models"""
    return get_tracker().get_completed_classification_models()
