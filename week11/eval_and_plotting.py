import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
from add_noise import noisy_img
from model_tracking import get_tracker
import torch.nn.functional as F


def get_optimizer_display(optimize_all_layers):
    """Helper function to get optimizer scope display text"""
    return "All Layers (Conv+Linear)" if optimize_all_layers else "Linear Only (fc1-3)"


def plot_training_metrics_with_seeds(model_names, save_dir="plots/aggregate_seed_analysis"):
    """
    FIXED: Plot training curves with error bars (not ribbons) across multiple seeds
    UPDATED: Metadata at bottom right outside plot area for ALL plots
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
    class_timesteps = config.get('timesteps', 0)
    dataset = config.get('Dataset', 'unknown')
    optimize_all_layers = config.get('optimize_all_layers', False)
    
    # Extract base model info for classification models
    base_model_name = config.get('base_recon_model', None)
    recon_timesteps = None
    recon_pattern = None
    recon_dataset = None
    
    if base_model_name:
        base_model_info = tracker.get_model(base_model_name)
        if base_model_info:
            base_config = base_model_info['config']
            recon_timesteps = base_config.get('timesteps', 'unknown')
            recon_pattern = base_config.get('pattern', 'unknown')
            recon_dataset = base_config.get('Dataset', 'unknown')
    
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
    print(f"Classification Timesteps: {class_timesteps}")
    if recon_timesteps:
        print(f"Reconstruction Timesteps: {recon_timesteps}")
    print(f"Dataset: {dataset}")
    if train_cond == "classification_training_shapes":
        print(f"Optimizer: {get_optimizer_display(optimize_all_layers)}")
    print(f"Seeds: {seeds_used}")
    print(f"Models: {len(model_names)}")
    
    # Base name now includes optimizer scope
    base_name = model_names[0].rsplit('_s', 1)[0]
    
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
        
        # Adaptive sampling based on number of epochs
        if min_epochs <= 25:
            error_bar_interval = 1
        elif min_epochs <= 50:
            error_bar_interval = 2
        elif min_epochs <= 100:
            error_bar_interval = 5
        else:
            error_bar_interval = max(5, min_epochs // 20)
        
        error_epochs = epochs[::error_bar_interval]
        error_mean_train = mean_train[::error_bar_interval]
        error_std_train = std_train[::error_bar_interval]
        error_mean_test = mean_test[::error_bar_interval]
        error_std_test = std_test[::error_bar_interval]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot lines
        ax.plot(epochs, mean_train, label='Train Loss', linewidth=2.5, color='#2E86DE')
        ax.plot(epochs, mean_test, label='Test Loss', linewidth=2.5, color='#EE5A6F')
        
        # Add error bars
        ax.errorbar(error_epochs, error_mean_train, yerr=error_std_train, 
                   fmt='none', ecolor='#2E86DE', alpha=0.4, capsize=3, capthick=1.5)
        ax.errorbar(error_epochs, error_mean_test, yerr=error_std_test, 
                   fmt='none', ecolor='#EE5A6F', alpha=0.4, capsize=3, capthick=1.5)
        
        # Main title
        ax.set_title('Reconstruction Training', fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with timesteps
        ax.text(0.5, 1.02, f'Reconstruction Timesteps: {class_timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=13, fontweight='bold')
        
        # Legend inside plot area (upper right)
        ax.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(alpha=0.3, linestyle='--')
        
        # ✅ Metadata at bottom right OUTSIDE plot area
        info_lines = [
            f'Pattern: {pattern}',
            f'Seeds: n={len(seeds_used)}',
            f'Dataset: {dataset_name}'
        ]
        info_text = '\n'.join(info_lines)
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
                ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
        
        # Adaptive sampling
        if min_epochs <= 25:
            error_bar_interval = 1
        elif min_epochs <= 50:
            error_bar_interval = 2
        elif min_epochs <= 100:
            error_bar_interval = 5
        else:
            error_bar_interval = max(5, min_epochs // 20)
        
        error_epochs = epochs[::error_bar_interval]
        error_mean_train = mean_train[::error_bar_interval]
        error_std_train = std_train[::error_bar_interval]
        error_mean_test = mean_test[::error_bar_interval]
        error_std_test = std_test[::error_bar_interval]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot lines
        ax.plot(epochs, mean_train, label='Train Loss', linewidth=2.5, color='#6C5CE7')
        ax.plot(epochs, mean_test, label='Test Loss', linewidth=2.5, color='#FD79A8')
        
        # Add error bars
        ax.errorbar(error_epochs, error_mean_train, yerr=error_std_train, 
                   fmt='none', ecolor='#6C5CE7', alpha=0.4, capsize=3, capthick=1.5)
        ax.errorbar(error_epochs, error_mean_test, yerr=error_std_test, 
                   fmt='none', ecolor='#FD79A8', alpha=0.4, capsize=3, capthick=1.5)
        
        # Main title
        ax.set_title('Classification Training', fontsize=16, fontweight='bold', pad=20)
        
        # Enhanced subtitle with BOTH timesteps
        if recon_timesteps:
            subtitle = f'Reconstruction: {recon_timesteps} steps | Classification: {class_timesteps} steps'
        else:
            subtitle = f'Classification Timesteps: {class_timesteps}'
        ax.text(0.5, 1.02, subtitle, 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Classification Loss (Cross-Entropy)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        # ✅ Metadata at bottom right OUTSIDE plot area
        info_lines = [
            f'Classification Pattern: {pattern}',
            f'Seeds: n={len(seeds_used)}',
            f'Train Dataset: {dataset_name}',
            f'Optimizer: {get_optimizer_display(optimize_all_layers)}'
        ]
        if recon_pattern and recon_dataset:
            recon_ds_name = dataset_display.get(recon_dataset, recon_dataset)
            info_lines.append(f'Base Model: {recon_pattern}, {recon_ds_name}')
        
        info_text = '\n'.join(info_lines)
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
                ha='left', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
        
        # Adaptive sampling
        if min_epochs <= 25:
            error_bar_interval = 1
        elif min_epochs <= 50:
            error_bar_interval = 2
        elif min_epochs <= 100:
            error_bar_interval = 5
        else:
            error_bar_interval = max(5, min_epochs // 20)
        
        error_epochs = epochs[::error_bar_interval]
        error_mean_train = mean_train_acc[::error_bar_interval]
        error_std_train = std_train_acc[::error_bar_interval]
        error_mean_test = mean_test_acc[::error_bar_interval]
        error_std_test = std_test_acc[::error_bar_interval]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot lines
        ax.plot(epochs, mean_train_acc, label='Train Accuracy', linewidth=2.5, color='#00B894')
        ax.plot(epochs, mean_test_acc, label='Test Accuracy', linewidth=2.5, color='#FDCB6E')
        
        # Add error bars
        ax.errorbar(error_epochs, error_mean_train, yerr=error_std_train, 
                   fmt='none', ecolor='#00B894', alpha=0.4, capsize=3, capthick=1.5)
        ax.errorbar(error_epochs, error_mean_test, yerr=error_std_test, 
                   fmt='none', ecolor='#FDCB6E', alpha=0.4, capsize=3, capthick=1.5)
        
        # Main title
        ax.set_title('Classification Training', fontsize=16, fontweight='bold', pad=20)
        
        # Enhanced subtitle
        if recon_timesteps:
            subtitle = f'Reconstruction: {recon_timesteps} steps | Classification: {class_timesteps} steps'
        else:
            subtitle = f'Classification Timesteps: {class_timesteps}'
        ax.text(0.5, 1.02, subtitle, 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        # ✅ Metadata at bottom right OUTSIDE plot area
        info_lines = [
            f'Classification Pattern: {pattern}',
            f'Seeds: n={len(seeds_used)}',
            f'Train Dataset: {dataset_name}',
            f'Optimizer: {get_optimizer_display(optimize_all_layers)}'
        ]
        if recon_pattern and recon_dataset:
            recon_ds_name = dataset_display.get(recon_dataset, recon_dataset)
            info_lines.append(f'Base Model: {recon_pattern}, {recon_ds_name}')
        
        info_text = '\n'.join(info_lines)
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
                ha='left', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
        
        # Adaptive sampling
        if min_epochs <= 25:
            error_bar_interval = 1
        elif min_epochs <= 50:
            error_bar_interval = 2
        elif min_epochs <= 100:
            error_bar_interval = 5
        else:
            error_bar_interval = max(5, min_epochs // 20)
        
        error_epochs = epochs[::error_bar_interval]
        error_mean_illusion = mean_illusion[::error_bar_interval]
        error_std_illusion = std_illusion[::error_bar_interval]
        error_mean_cifar = mean_cifar[::error_bar_interval]
        error_std_cifar = std_cifar[::error_bar_interval]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot lines
        ax.plot(epochs, mean_illusion, label='Illusion Dataset', linewidth=2.5, color='#E17055')
        ax.plot(epochs, mean_cifar, label='CIFAR-10 Dataset', linewidth=2.5, color='#74B9FF')
        
        # Add error bars
        ax.errorbar(error_epochs, error_mean_illusion, yerr=error_std_illusion, 
                   fmt='none', ecolor='#E17055', alpha=0.4, capsize=3, capthick=1.5)
        ax.errorbar(error_epochs, error_mean_cifar, yerr=error_std_cifar, 
                   fmt='none', ecolor='#74B9FF', alpha=0.4, capsize=3, capthick=1.5)
        
        # Main title
        ax.set_title('Classification Training', fontsize=16, fontweight='bold', pad=20)
        
        # Enhanced subtitle
        if recon_timesteps:
            subtitle = f'Reconstruction: {recon_timesteps} steps | Classification: {class_timesteps} steps'
        else:
            subtitle = f'Classification Timesteps: {class_timesteps}'
        ax.text(0.5, 1.02, subtitle, 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        # ✅ Metadata at bottom right OUTSIDE plot area
        info_lines = [
            f'Classification Pattern: {pattern}',
            f'Seeds: n={len(seeds_used)}',
            f'Train: {dataset_name} | Test: CIFAR-10 & Illusion',
            f'Optimizer: {get_optimizer_display(optimize_all_layers)}'
        ]
        if recon_pattern and recon_dataset:
            recon_ds_name = dataset_display.get(recon_dataset, recon_dataset)
            info_lines.append(f'Base Model: {recon_pattern}, {recon_ds_name}')
        
        info_text = '\n'.join(info_lines)
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
                ha='left', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        filename = f"{base_name}_ReconstructionComparison.png"
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")
    
    print(f"{'='*60}\n")


def plot_training_metrics(metrics_history, model_name, config):
    """
    FIXED: Plot training metrics for individual models with unique filenames
    UPDATED: Metadata at bottom right outside plot for ALL plots
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
    timesteps = getattr(config, 'timesteps', 0)
    dataset = getattr(config, 'classification_datasetpath', 'unknown')
    optimize_all_layers = getattr(config, 'optimize_all_layers', False)
    
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
            ax.text(0.5, 1.02, f'Reconstruction Timesteps: {timesteps}', 
                    transform=ax.transAxes, ha='center', fontsize=11, style='italic')
            ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=13, fontweight='bold')
            metric_type = "ReconstructionLoss"
            
            # ✅ Metadata at bottom right OUTSIDE plot
            info_lines = [
                f'Pattern: {pattern}',
                f'Seed: {seed}',
                f'Dataset: {dataset_name}'
            ]
        else:
            # Classification training
            ax.set_title("Classification Training", fontsize=16, fontweight='bold', pad=20)
            ax.text(0.5, 1.02, f'Classification Timesteps: {timesteps}', 
                    transform=ax.transAxes, ha='center', fontsize=11, style='italic')
            ax.set_ylabel("Classification Loss (Cross-Entropy)", fontsize=13, fontweight='bold')
            metric_type = "ClassificationLoss"
            
            # ✅ Metadata at bottom right OUTSIDE plot
            info_lines = [
                f'Pattern: {pattern}',
                f'Seed: {seed}',
                f'Dataset: {dataset_name}',
                f'Optimizer: {get_optimizer_display(optimize_all_layers)}'
            ]
        
        info_text = '\n'.join(info_lines)
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
                ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

        ax.set_title("Classification Training", fontsize=16, fontweight='bold', pad=20)
        ax.text(0.5, 1.02, f'Classification Timesteps: {timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # ✅ Metadata at bottom right OUTSIDE plot
        info_lines = [
            f'Pattern: {pattern}',
            f'Seed: {seed}',
            f'Dataset: {dataset_name}',
            f'Optimizer: {get_optimizer_display(optimize_all_layers)}'
        ]
        info_text = '\n'.join(info_lines)
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
                ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

        ax.set_title("Classification Training", fontsize=16, fontweight='bold', pad=20)
        ax.text(0.5, 1.02, f'Classification Timesteps: {timesteps}', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # ✅ Metadata at bottom right OUTSIDE plot
        info_lines = [
            f'Pattern: {pattern}',
            f'Seed: {seed}',
            f'Train: {dataset_name} | Test: CIFAR-10 & Illusion',
            f'Optimizer: {get_optimizer_display(optimize_all_layers)}'
        ]
        info_text = '\n'.join(info_lines)
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
                ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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


def plot_test_trajectory(class_results, model_name, config):
    """
    Plot trajectory of illusion perception across timesteps
    UPDATED: Metadata at bottom right outside plot
    """
    os.makedirs("plots/test_trajectories", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f'PC Dynamics Trajectory\nModel: {model_name}', fontsize=12)

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

        ax.plot(
            timesteps,
            mean_probs,
            linewidth=2,
            marker='o',
            markersize=5,
            label=cls_name.replace('_', ' ').title(),
            color=colors.get(cls_name, '#95a5a6')
        )

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Probability of being correct (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/test_trajectories/{model_name}_trajectory.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


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
