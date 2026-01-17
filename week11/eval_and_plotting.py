import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
from add_noise import noisy_img
from model_tracking import get_tracker


def plot_training_metrics_with_seeds(model_names, save_dir="plots/aggregate_seed_analysis"):
    """
    FIXED: Plot training curves with error bars across multiple seeds
    Filenames: {model_name}_{metric_type}.png
    Enhanced plot info without clutter
    """
    os.makedirs(save_dir, exist_ok=True)
    
    tracker = get_tracker()
    
    print(f"\n{'='*60}")
    print(f"PLOTTING AGGREGATE METRICS ACROSS SEEDS")
    print(f"{'='*60}")
    
    # Get config info
    first_model = tracker.get_model(model_names[0])
    if not first_model:
        print("⚠ No model info found")
        return
    
    config = first_model['config']
    pattern = config.get('pattern', 'unknown')
    train_cond = config.get('train_cond', 'unknown')
    recon_timesteps = config.get('timesteps', 0)
    dataset = config.get('Dataset', 'unknown')
    
    # Collect metrics from all seeds
    all_train_loss = []
    all_test_loss = []
    all_train_acc = []
    all_test_acc = []
    all_illusion_recon = []
    all_cifar_recon = []
    seeds_used = []
    
    for model_name in model_names:
        model_info = tracker.get_model(model_name)
        if not model_info:
            continue
        
        metrics = model_info.get('metrics', {})
        model_config = model_info.get('config', {})
        seed = model_config.get('seed', 'unknown')
        
        if 'train_loss' in metrics and len(metrics['train_loss']) > 0:
            all_train_loss.append(metrics['train_loss'])
            all_test_loss.append(metrics['test_loss'])
            seeds_used.append(seed)
            
            if 'train_acc' in metrics:
                all_train_acc.append(metrics['train_acc'])
                all_test_acc.append(metrics['test_acc'])
            
            if 'illusory_datset_recon_loss' in metrics:
                all_illusion_recon.append(metrics['illusory_datset_recon_loss'])
                all_cifar_recon.append(metrics['cifar10_dataset_recon_loss'])
    
    if not all_train_loss:
        print("⚠ No training metrics found")
        return
    
    print(f"Pattern: {pattern}")
    print(f"Condition: {train_cond}")
    print(f"Timesteps: {recon_timesteps}")
    print(f"Dataset: {dataset}")
    print(f"Seeds: {seeds_used}")
    print(f"Models: {len(model_names)}")
    
    # Base model name (remove seed suffix for filename)
    base_name = model_names[0].rsplit('_s', 1)[0]  # Remove _s{seed} suffix
    
    # Dataset name mapping
    dataset_display = {
        "cifar10": "CIFAR-10",
        "stl10": "STL-10",
        "custom_illusion_dataset": "Illusion"
    }
    dataset_name = dataset_display.get(dataset, dataset)
    
    # ================================================================
    # PLOT 1: RECONSTRUCTION LOSS (for recon models only)
    # ================================================================
    if train_cond == "recon_pc_train":
        min_epochs = min(len(x) for x in all_train_loss)
        all_train_loss_arr = np.array([x[:min_epochs] for x in all_train_loss])
        all_test_loss_arr = np.array([x[:min_epochs] for x in all_test_loss])
        
        mean_train = np.mean(all_train_loss_arr, axis=0)
        std_train = np.std(all_train_loss_arr, axis=0)
        mean_test = np.mean(all_test_loss_arr, axis=0)
        std_test = np.std(all_test_loss_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_train, label='Train Loss', linewidth=2.5, color='#2E86DE')
        ax.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                        alpha=0.25, color='#2E86DE')
        
        ax.plot(epochs, mean_test, label='Test Loss', linewidth=2.5, color='#EE5A6F')
        ax.fill_between(epochs, mean_test - std_test, mean_test + std_test, 
                        alpha=0.25, color='#EE5A6F')
        
        # Main title
        ax.set_title('Reconstruction Training', fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with timesteps
        ax.text(0.5, 1.02, f'Reconstruction Timesteps: {recon_timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Bottom right info box
        info_text = f'Pattern: {pattern}\nSeeds: n={len(seeds_used)}\nDataset: {dataset_name}'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        filename = f"{base_name}_ReconstructionLoss.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")
    
    # ================================================================
    # PLOT 2: CLASSIFICATION LOSS (for classification models only)
    # ================================================================
    if train_cond == "classification_training_shapes":
        min_epochs = min(len(x) for x in all_train_loss)
        all_train_loss_arr = np.array([x[:min_epochs] for x in all_train_loss])
        all_test_loss_arr = np.array([x[:min_epochs] for x in all_test_loss])
        
        mean_train = np.mean(all_train_loss_arr, axis=0)
        std_train = np.std(all_train_loss_arr, axis=0)
        mean_test = np.mean(all_test_loss_arr, axis=0)
        std_test = np.std(all_test_loss_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        
        # Extract classification timesteps from model name
        class_timesteps = recon_timesteps  # Default assumption
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_train, label='Train Loss', linewidth=2.5, color='#6C5CE7')
        ax.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                        alpha=0.25, color='#6C5CE7')
        
        ax.plot(epochs, mean_test, label='Test Loss', linewidth=2.5, color='#FD79A8')
        ax.fill_between(epochs, mean_test - std_test, mean_test + std_test, 
                        alpha=0.25, color='#FD79A8')
        
        # Main title
        ax.set_title('Classification Training', fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with timesteps
        ax.text(0.5, 1.02, f'Classification Timesteps: {class_timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Bottom right info box
        info_text = f'Pattern: {pattern}\nSeeds: n={len(seeds_used)}\nDataset: {dataset_name}'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Classification Loss (Cross-Entropy)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        filename = f"{base_name}_ClassificationLoss.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")
    
    # ================================================================
    # PLOT 3: CLASSIFICATION ACCURACY (for classification models only)
    # ================================================================
    if all_train_acc:
        min_epochs = min(len(x) for x in all_train_acc)
        all_train_acc_arr = np.array([x[:min_epochs] for x in all_train_acc])
        all_test_acc_arr = np.array([x[:min_epochs] for x in all_test_acc])
        
        mean_train_acc = np.mean(all_train_acc_arr, axis=0)
        std_train_acc = np.std(all_train_acc_arr, axis=0)
        mean_test_acc = np.mean(all_test_acc_arr, axis=0)
        std_test_acc = np.std(all_test_acc_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        class_timesteps = recon_timesteps
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_train_acc, label='Train Accuracy', linewidth=2.5, color='#00B894')
        ax.fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, 
                        alpha=0.25, color='#00B894')
        
        ax.plot(epochs, mean_test_acc, label='Test Accuracy', linewidth=2.5, color='#FDCB6E')
        ax.fill_between(epochs, mean_test_acc - std_test_acc, mean_test_acc + std_test_acc, 
                        alpha=0.25, color='#FDCB6E')
        
        # Main title
        ax.set_title('Classification Training', fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with timesteps
        ax.text(0.5, 1.02, f'Classification Timesteps: {class_timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Bottom right info box
        info_text = f'Pattern: {pattern}\nSeeds: n={len(seeds_used)}\nDataset: {dataset_name}'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        filename = f"{base_name}_ClassificationAccuracy.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")
    
    # ================================================================
    # PLOT 4: RECONSTRUCTION LOSS COMPARISON (for classification models)
    # ================================================================
    if all_illusion_recon and all_cifar_recon:
        min_epochs = min(len(x) for x in all_illusion_recon)
        all_illusion_arr = np.array([x[:min_epochs] for x in all_illusion_recon])
        all_cifar_arr = np.array([x[:min_epochs] for x in all_cifar_recon])
        
        mean_illusion = np.mean(all_illusion_arr, axis=0)
        std_illusion = np.std(all_illusion_arr, axis=0)
        mean_cifar = np.mean(all_cifar_arr, axis=0)
        std_cifar = np.std(all_cifar_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        class_timesteps = recon_timesteps
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_illusion, label='Illusion Dataset', linewidth=2.5, color='#E17055')
        ax.fill_between(epochs, mean_illusion - std_illusion, mean_illusion + std_illusion, 
                        alpha=0.25, color='#E17055')
        
        ax.plot(epochs, mean_cifar, label='CIFAR-10 Dataset', linewidth=2.5, color='#74B9FF')
        ax.fill_between(epochs, mean_cifar - std_cifar, mean_cifar + std_cifar, 
                        alpha=0.25, color='#74B9FF')
        
        # Main title
        ax.set_title('Classification Training', fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with timesteps
        ax.text(0.5, 1.02, f'Classification Timesteps: {class_timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Bottom right info box
        info_text = f'Pattern: {pattern}\nSeeds: n={len(seeds_used)}\nTrain: {dataset_name} | Test: CIFAR-10 & Illusion'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        filename = f"{base_name}_ReconstructionComparison.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")
    
    print(f"{'='*60}\n")


def plot_training_metrics(metrics_history, model_name, config):
    """
    Plot training metrics for individual models
    FIXED: Simple filenames with enhanced plot information
    """
    os.makedirs("plots/individual_training_metrics", exist_ok=True)

    if not metrics_history.get("train_loss"):
        return

    epochs = range(1, len(metrics_history["train_loss"]) + 1)
    
    # Extract config info
    pattern = "Unknown"
    if hasattr(config, 'gammaset') and config.gammaset:
        gamma_vals = config.gammaset[0]
        if all(abs(g - 0.33) < 0.01 for g in gamma_vals):
            pattern = "Uniform"
        elif gamma_vals[0] < gamma_vals[2]:
            pattern = "Gamma Increasing"
        elif gamma_vals[0] > gamma_vals[2]:
            pattern = "Gamma Decreasing"
        else:
            beta_vals = config.betaset[0] if hasattr(config, 'betaset') else []
            if beta_vals and beta_vals[0] < beta_vals[2]:
                pattern = "Beta Increasing"
            elif beta_vals and beta_vals[0] > beta_vals[2]:
                pattern = "Beta Decreasing"
    
    seed = getattr(config, 'seed', 0)
    recon_timesteps = getattr(config, 'timesteps', 0)
    dataset = getattr(config, 'classification_datasetpath', 'unknown')
    
    # Dataset display name
    dataset_display = {
        "cifar10": "CIFAR-10",
        "stl10": "STL-10",
        "custom_illusion_dataset": "Illusion"
    }
    dataset_name = dataset_display.get(dataset, dataset)
    
    # ================================================================
    # PLOT 1: Loss (Reconstruction or Classification)
    # ================================================================
    if "train_loss" in metrics_history and "test_loss" in metrics_history:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epochs, metrics_history["train_loss"], label="Train Loss", linewidth=2.5, color='#3498db')
        ax.plot(epochs, metrics_history["test_loss"], label="Test Loss", linewidth=2.5, color='#e74c3c')

        if config.training_condition == "recon_pc_train":
            # Reconstruction training
            ax.set_title("Reconstruction Training", fontsize=16, fontweight='bold', pad=20)
            ax.text(0.5, 1.02, f'Reconstruction Timesteps: {recon_timesteps}', 
                    transform=ax.transAxes, ha='center', fontsize=11, style='italic')
            ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=13, fontweight='bold')
            metric_type = "ReconstructionLoss"
        else:
            # Classification training
            ax.set_title("Classification Training", fontsize=16, fontweight='bold', pad=20)
            ax.text(0.5, 1.02, f'Classification Timesteps: {recon_timesteps}', 
                    transform=ax.transAxes, ha='center', fontsize=11, style='italic')
            ax.set_ylabel("Classification Loss (Cross-Entropy)", fontsize=13, fontweight='bold')
            metric_type = "ClassificationLoss"
        
        # Bottom right info box
        info_text = f'Pattern: {pattern}\nSeed: {seed}\nDataset: {dataset_name}'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')

        filename = f"{model_name}_{metric_type}.png"
        save_path = f"plots/individual_training_metrics/{filename}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    # ================================================================
    # PLOT 2: Classification Accuracy
    # ================================================================
    if "train_acc" in metrics_history and "test_acc" in metrics_history:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epochs, metrics_history["train_acc"], label="Train Accuracy", linewidth=2.5, color='#2ecc71')
        ax.plot(epochs, metrics_history["test_acc"], label="Test Accuracy", linewidth=2.5, color='#f39c12')

        # Main title
        ax.set_title("Classification Training", fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with timesteps
        ax.text(0.5, 1.02, f'Classification Timesteps: {recon_timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Bottom right info box
        info_text = f'Pattern: {pattern}\nSeed: {seed}\nDataset: {dataset_name}'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])

        filename = f"{model_name}_ClassificationAccuracy.png"
        save_path = f"plots/individual_training_metrics/{filename}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    # ================================================================
    # PLOT 3: Reconstruction Loss Comparison
    # ================================================================
    if "illusory_datset_recon_loss" in metrics_history and "cifar10_dataset_recon_loss" in metrics_history:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epochs, metrics_history["illusory_datset_recon_loss"],
                 label="Illusion Dataset", linewidth=2.5, color='#9b59b6')
        ax.plot(epochs, metrics_history["cifar10_dataset_recon_loss"],
                 label="CIFAR-10 Dataset", linewidth=2.5, color='#1abc9c')

        # Main title
        ax.set_title("Classification Training", fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with timesteps
        ax.text(0.5, 1.02, f'Classification Timesteps: {recon_timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Bottom right info box
        info_text = f'Pattern: {pattern}\nSeed: {seed}\nTrain: {dataset_name} | Test: CIFAR-10 & Illusion'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
        ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')

        filename = f"{model_name}_ReconstructionComparison.png"
        save_path = f"plots/individual_training_metrics/{filename}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")
    """
    FIXED: Plot training curves with error bars across multiple seeds
    Creates SEPARATE plots for:
    - Reconstruction loss (for recon models)
    - Classification loss (for classification models)
    - Classification accuracy (for classification models)
    
    Args:
        model_names: List of model names with same config but different seeds
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    tracker = get_tracker()
    
    print(f"\n{'='*60}")
    print(f"PLOTTING AGGREGATE METRICS ACROSS SEEDS")
    print(f"{'='*60}")
    
    # Get config info for naming
    first_model = tracker.get_model(model_names[0])
    if not first_model:
        print("⚠ No model info found")
        return
    
    config = first_model['config']
    pattern = config.get('pattern', 'unknown')
    train_cond = config.get('train_cond', 'unknown')
    timesteps = config.get('timesteps', 0)
    dataset = config.get('Dataset', 'unknown')
    lr = config.get('lr', 0)
    
    # Collect metrics from all seeds
    all_train_loss = []
    all_test_loss = []
    all_train_acc = []
    all_test_acc = []
    all_illusion_recon = []
    all_cifar_recon = []
    seeds_used = []
    
    for model_name in model_names:
        model_info = tracker.get_model(model_name)
        if not model_info:
            continue
        
        metrics = model_info.get('metrics', {})
        model_config = model_info.get('config', {})
        seed = model_config.get('seed', 'unknown')
        
        if 'train_loss' in metrics and len(metrics['train_loss']) > 0:
            all_train_loss.append(metrics['train_loss'])
            all_test_loss.append(metrics['test_loss'])
            seeds_used.append(seed)
            
            if 'train_acc' in metrics:
                all_train_acc.append(metrics['train_acc'])
                all_test_acc.append(metrics['test_acc'])
            
            if 'illusory_datset_recon_loss' in metrics:
                all_illusion_recon.append(metrics['illusory_datset_recon_loss'])
                all_cifar_recon.append(metrics['cifar10_dataset_recon_loss'])
    
    if not all_train_loss:
        print("⚠ No training metrics found")
        return
    
    print(f"Pattern: {pattern}")
    print(f"Condition: {train_cond}")
    print(f"Timesteps: {timesteps}")
    print(f"Dataset: {dataset}")
    print(f"Learning rate: {lr}")
    print(f"Seeds: {seeds_used}")
    print(f"Models: {len(model_names)}")
    
    # Dataset name mapping for filenames
    dataset_names = {
        "cifar10": "CIFAR10",
        "stl10": "STL10",
        "custom_illusion_dataset": "IllusionDataset"
    }
    dataset_full = dataset_names.get(dataset, dataset)
    
    # Training condition names
    cond_names = {
        "recon_pc_train": "ReconstructionTraining",
        "classification_training_shapes": "ClassificationTraining"
    }
    cond_full = cond_names.get(train_cond, train_cond)
    
    # ================================================================
    # PLOT 1: RECONSTRUCTION LOSS (for recon models only)
    # ================================================================
    if train_cond == "recon_pc_train":
        min_epochs = min(len(x) for x in all_train_loss)
        all_train_loss_arr = np.array([x[:min_epochs] for x in all_train_loss])
        all_test_loss_arr = np.array([x[:min_epochs] for x in all_test_loss])
        
        mean_train = np.mean(all_train_loss_arr, axis=0)
        std_train = np.std(all_train_loss_arr, axis=0)
        mean_test = np.mean(all_test_loss_arr, axis=0)
        std_test = np.std(all_test_loss_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_train, label='Train Reconstruction Loss', linewidth=2.5, color='#2E86DE')
        ax.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                        alpha=0.25, color='#2E86DE')
        
        ax.plot(epochs, mean_test, label='Test Reconstruction Loss', linewidth=2.5, color='#EE5A6F')
        ax.fill_between(epochs, mean_test - std_test, mean_test + std_test, 
                        alpha=0.25, color='#EE5A6F')
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
        ax.set_title(f'Reconstruction Loss: {pattern} Pattern on {dataset_full}\n'
                    f'Timesteps={timesteps}, LR={lr}, n={len(seeds_used)} seeds {seeds_used}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        filename = f"ReconLoss_{pattern}_{dataset_full}_t{timesteps}_lr{lr}_nseeds{len(seeds_used)}.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved reconstruction loss: {filepath}")
    
    # ================================================================
    # PLOT 2: CLASSIFICATION LOSS (for classification models only)
    # ================================================================
    if train_cond == "classification_training_shapes":
        min_epochs = min(len(x) for x in all_train_loss)
        all_train_loss_arr = np.array([x[:min_epochs] for x in all_train_loss])
        all_test_loss_arr = np.array([x[:min_epochs] for x in all_test_loss])
        
        mean_train = np.mean(all_train_loss_arr, axis=0)
        std_train = np.std(all_train_loss_arr, axis=0)
        mean_test = np.mean(all_test_loss_arr, axis=0)
        std_test = np.std(all_test_loss_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_train, label='Train Classification Loss', linewidth=2.5, color='#6C5CE7')
        ax.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                        alpha=0.25, color='#6C5CE7')
        
        ax.plot(epochs, mean_test, label='Test Classification Loss', linewidth=2.5, color='#FD79A8')
        ax.fill_between(epochs, mean_test - std_test, mean_test + std_test, 
                        alpha=0.25, color='#FD79A8')
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Classification Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
        ax.set_title(f'Classification Loss: {pattern} Pattern on {dataset_full}\n'
                    f'Timesteps={timesteps}, LR={lr}, n={len(seeds_used)} seeds {seeds_used}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        filename = f"ClassLoss_{pattern}_{dataset_full}_t{timesteps}_lr{lr}_nseeds{len(seeds_used)}.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved classification loss: {filepath}")
    
    # ================================================================
    # PLOT 3: CLASSIFICATION ACCURACY (for classification models only)
    # ================================================================
    if all_train_acc:
        min_epochs = min(len(x) for x in all_train_acc)
        all_train_acc_arr = np.array([x[:min_epochs] for x in all_train_acc])
        all_test_acc_arr = np.array([x[:min_epochs] for x in all_test_acc])
        
        mean_train_acc = np.mean(all_train_acc_arr, axis=0)
        std_train_acc = np.std(all_train_acc_arr, axis=0)
        mean_test_acc = np.mean(all_test_acc_arr, axis=0)
        std_test_acc = np.std(all_test_acc_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_train_acc, label='Train Accuracy', linewidth=2.5, color='#00B894')
        ax.fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, 
                        alpha=0.25, color='#00B894')
        
        ax.plot(epochs, mean_test_acc, label='Test Accuracy', linewidth=2.5, color='#FDCB6E')
        ax.fill_between(epochs, mean_test_acc - std_test_acc, mean_test_acc + std_test_acc, 
                        alpha=0.25, color='#FDCB6E')
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Classification Accuracy: {pattern} Pattern on {dataset_full}\n'
                    f'Timesteps={timesteps}, LR={lr}, n={len(seeds_used)} seeds {seeds_used}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        filename = f"ClassAccuracy_{pattern}_{dataset_full}_t{timesteps}_lr{lr}_nseeds{len(seeds_used)}.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved classification accuracy: {filepath}")
    
    # ================================================================
    # PLOT 4: RECONSTRUCTION LOSS COMPARISON (for classification models)
    # ================================================================
    if all_illusion_recon and all_cifar_recon:
        min_epochs = min(len(x) for x in all_illusion_recon)
        all_illusion_arr = np.array([x[:min_epochs] for x in all_illusion_recon])
        all_cifar_arr = np.array([x[:min_epochs] for x in all_cifar_recon])
        
        mean_illusion = np.mean(all_illusion_arr, axis=0)
        std_illusion = np.std(all_illusion_arr, axis=0)
        mean_cifar = np.mean(all_cifar_arr, axis=0)
        std_cifar = np.std(all_cifar_arr, axis=0)
        
        epochs = np.arange(1, min_epochs + 1)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(epochs, mean_illusion, label='Illusion Dataset Recon Loss', linewidth=2.5, color='#E17055')
        ax.fill_between(epochs, mean_illusion - std_illusion, mean_illusion + std_illusion, 
                        alpha=0.25, color='#E17055')
        
        ax.plot(epochs, mean_cifar, label='CIFAR10 Dataset Recon Loss', linewidth=2.5, color='#74B9FF')
        ax.fill_between(epochs, mean_cifar - std_cifar, mean_cifar + std_cifar, 
                        alpha=0.25, color='#74B9FF')
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
        ax.set_title(f'Reconstruction Loss Comparison: {pattern} Pattern\n'
                    f'Timesteps={timesteps}, LR={lr}, n={len(seeds_used)} seeds {seeds_used}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        filename = f"ReconComparison_{pattern}_{dataset_full}_t{timesteps}_lr{lr}_nseeds{len(seeds_used)}.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved reconstruction comparison: {filepath}")
    
    print(f"{'='*60}\n")


def plot_test_trajectory(class_results, model_name, config):
    """
    Plot trajectory of illusion perception across timesteps
    All classes shown in a single plot
    """
    os.makedirs("plots/test_trajectories", exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.title(f'Pc Dynamics Trajectory\nModel: {model_name}', fontsize=12)

    colors = {
        'square': '#2ecc71',
        'rectangle': '#3498db',
        'trapezium': '#e74c3c',
        'triangle': '#9b59b6',
        'hexagon': '#f39c12',
        'random': '#34495e',
        'all_in': '#e67e22',
        'all_out': '#1abc9c'
    }

    for cls_name, cls_data in class_results.items():
        if cls_data["total"] == 0:
            continue

        mean_probs = [
            np.mean(p) * 100 if len(p) > 0 else 0.0
            for p in cls_data["predictions"]
        ]

        timesteps = range(len(mean_probs))

        plt.plot(
            timesteps,
            mean_probs,
            linewidth=2,
            marker='o',
            markersize=5,
            label=cls_name.replace('_', ' ').title(),
            color=colors.get(cls_name, '#95a5a6')
        )

    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Probability of being correct (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/test_trajectories/{model_name}_trajectory_t{config.timesteps}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_metrics(metrics_history, model_name, config):
    """
    Plot training metrics for individual models
    FIXED: Better filenames that describe the plot content
    """
    os.makedirs("plots/individual_training_metrics", exist_ok=True)

    if not metrics_history.get("train_loss"):
        return

    epochs = range(1, len(metrics_history["train_loss"]) + 1)
    
    # Extract config info for better filenames
    pattern = getattr(config, 'experiment_name', 'Unknown')
    if hasattr(config, 'gammaset') and config.gammaset:
        # Try to infer pattern from gamma values
        gamma_vals = config.gammaset[0]
        if all(abs(g - 0.33) < 0.01 for g in gamma_vals):
            pattern_name = "Uniform"
        elif gamma_vals[0] < gamma_vals[2]:
            pattern_name = "GammaIncreasing"
        elif gamma_vals[0] > gamma_vals[2]:
            pattern_name = "GammaDecreasing"
        else:
            pattern_name = "CustomPattern"
    else:
        pattern_name = "UnknownPattern"
    
    seed = getattr(config, 'seed', 0)
    timesteps = getattr(config, 'timesteps', 0)
    
    # ================================================================
    # PLOT 1: Loss (Reconstruction or Classification)
    # ================================================================
    if "train_loss" in metrics_history and "test_loss" in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics_history["train_loss"], label="Train Loss", linewidth=2, color='#3498db')
        plt.plot(epochs, metrics_history["test_loss"], label="Test Loss", linewidth=2, color='#e74c3c')

        plt.xlabel("Epoch", fontsize=12)
        
        if config.training_condition == "recon_pc_train":
            plt.ylabel("Reconstruction Loss (MSE)", fontsize=12)
            plt.title(f"Reconstruction Loss\n{model_name}", fontsize=12, fontweight='bold')
            loss_type = "ReconLoss"
        else:
            plt.ylabel("Classification Loss (Cross-Entropy)", fontsize=12)
            plt.title(f"Classification Loss\n{model_name}", fontsize=12, fontweight='bold')
            loss_type = "ClassLoss"

        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)

        filename = f"{loss_type}_{pattern_name}_t{timesteps}_seed{seed}.png"
        save_path = f"plots/individual_training_metrics/{filename}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    # ================================================================
    # PLOT 2: Classification Accuracy
    # ================================================================
    if "train_acc" in metrics_history and "test_acc" in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics_history["train_acc"], label="Train Accuracy", linewidth=2, color='#2ecc71')
        plt.plot(epochs, metrics_history["test_acc"], label="Test Accuracy", linewidth=2, color='#f39c12')

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title(f"Classification Accuracy\n{model_name}", fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.ylim([0, 100])

        filename = f"Accuracy_{pattern_name}_t{timesteps}_seed{seed}.png"
        save_path = f"plots/individual_training_metrics/{filename}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    # ================================================================
    # PLOT 3: Reconstruction Loss Comparison (Illusion vs CIFAR10)
    # ================================================================
    if "illusory_datset_recon_loss" in metrics_history and "cifar10_dataset_recon_loss" in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics_history["illusory_datset_recon_loss"],
                 label="Illusion Dataset", linewidth=2, color='#9b59b6')
        plt.plot(epochs, metrics_history["cifar10_dataset_recon_loss"],
                 label="CIFAR-10 Dataset", linewidth=2, color='#1abc9c')

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Reconstruction Loss (MSE)", fontsize=12)
        plt.title(f"Reconstruction Loss Comparison\n{model_name}", fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)

        filename = f"ReconComparison_{pattern_name}_t{timesteps}_seed{seed}.png"
        save_path = f"plots/individual_training_metrics/{filename}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")


def recon_pc_loss(net, dataloader, config):
    """Calculate reconstruction loss on a dataset"""
    total_loss = []
    
    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch[:2]
        images, labels = images.to(config.device), labels.to(config.device)
        
        _, _, height, width = images.shape
        batch_size = images.size(0)
        
        ft_AB_pc_temp = torch.zeros(batch_size, 6, height, width, device=config.device)
        ft_BC_pc_temp = torch.zeros(batch_size, 16, height // 2, width // 2, device=config.device)
        ft_CD_pc_temp = torch.zeros(batch_size, 32, height // 4, width // 4, device=config.device)
        ft_DE_pc_temp = torch.zeros(batch_size, 128, height // 8, width // 8, device=config.device)

        ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp = net.feedforward_pass_no_dense(
            images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp
        )

        ft_AB_pc_temp.requires_grad_(True)
        ft_BC_pc_temp.requires_grad_(True)
        ft_CD_pc_temp.requires_grad_(True)
        ft_DE_pc_temp.requires_grad_(True)

        final_loss = 0

        for i in range(config.timesteps):
            ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, loss_of_layers = net.recon_predictive_coding_pass(
                images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,
                config.betaset, config.gammaset, config.alphaset, images.size(0)
            )
            final_loss += loss_of_layers

        final_loss = final_loss / config.timesteps
        total_loss.append(final_loss.item())
        
        del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, loss_of_layers, final_loss
        torch.cuda.empty_cache()

    test_loss = np.mean(total_loss)
    return test_loss


def eval_pc_ill_accuracy(net, dataloader, config, criterion):
    """Evaluate accuracy on illusion dataset"""
    total_correct = np.zeros(config.timesteps + 1)
    total_samples = 0
    running_loss = []
    val_recon_loss = []
    
    for images, labels, _, _ in dataloader:
        images, labels = images.to(config.device), labels.to(config.device)

        for noise in np.arange(0, 0.35, 0.05):
            images_noisy = noisy_img(images, "gauss", round(noise, 2))

            _, _, height, width = images_noisy.shape
            batch_size = images_noisy.size(0)
            
            ft_AB_pc_temp = torch.zeros(batch_size, 6, height, width, device=config.device)
            ft_BC_pc_temp = torch.zeros(batch_size, 16, height // 2, width // 2, device=config.device)
            ft_CD_pc_temp = torch.zeros(batch_size, 32, height // 4, width // 4, device=config.device)
            ft_DE_pc_temp = torch.zeros(batch_size, 128, height // 8, width // 8, device=config.device)

            ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
            ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                images_noisy, ft_AB_pc_temp, ft_BC_pc_temp,
                ft_CD_pc_temp, ft_DE_pc_temp
            )

            ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
            ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)

            _, predicted = torch.max(output, 1)
            total_correct[0] += (predicted == labels).sum().item()

            final_loss = 0
            recon_loss = 0

            for i in range(config.timesteps):
                output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(
                    images_noisy, ft_AB_pc_temp, ft_BC_pc_temp,
                    ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp,
                    config.betaset, config.gammaset, config.alphaset,
                    images_noisy.size(0)
                )

                loss = criterion(output, labels)
                final_loss += loss
                recon_loss += (loss_of_layers / 4.0)

                _, predicted = torch.max(output, 1)
                total_correct[i + 1] += (predicted == labels).sum().item()

            total_samples += labels.size(0)

            final_loss = final_loss / config.timesteps
            recon_loss = recon_loss / config.timesteps

            running_loss.append(final_loss.item())
            val_recon_loss.append(recon_loss.item())

    accuracy = [100 * c / total_samples for c in total_correct]
    accuracy = np.array(accuracy)

    mean_acc = np.mean(accuracy)
    test_loss = np.mean(running_loss)
    test_recon_loss = np.mean(val_recon_loss)

    return mean_acc, test_loss, test_recon_loss



