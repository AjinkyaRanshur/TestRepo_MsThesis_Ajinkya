import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from config import seed, device, batch_size, epochs, lr, momentum, timesteps,training_condition
import os
from PIL import Image
import json
from add_noise import noisy_img


def plot_test_trajectory(class_results, model_name, config):
    """
    Plot trajectory of illusion perception across timesteps
    All classes shown in a single plot
    """

    os.makedirs("plots/test_trajectories", exist_ok=True)

    # Create single figure + axis
    plt.figure(figsize=(10, 6))
    plt.title(f'Illusion Trajectory: {model_name}',
              fontsize=16, fontweight='bold')

    # Define colors
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

    # Plot ALL classes on the same axis
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
    plt.ylabel('Probability of being a shape(%)', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/test_trajectories/{model_name}_trajectory_timesteps{config.timesteps}.png", dpi=300)
    plt.show()


def plot_training_metrics(metrics_history, model_name, config):
    """
    Plot training metrics for classification models.
    Creates separate figures for:
    - classification loss
    - accuracy
    - reconstruction loss
    Only plots metrics that exist in metrics_history.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs("plots/training_metrics", exist_ok=True)

    epochs = range(1, len(metrics_history["train_loss"]) + 1)

    # --------------------------------------------------
    # 1. Classification Loss
    # --------------------------------------------------
    if "train_loss" in metrics_history and "test_loss" in metrics_history:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, metrics_history["train_loss"],
                 label="Train Loss", linewidth=2)
        plt.plot(epochs, metrics_history["test_loss"],
                 label="Test Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Classification Loss")
        plt.title(f"Classification Loss: {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)

        save_path = f"plots/training_metrics/{model_name}_loss.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # --------------------------------------------------
    # 2. Accuracy
    # --------------------------------------------------
    if "train_acc" in metrics_history and "test_acc" in metrics_history:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, metrics_history["train_acc"],
                 label="Train Accuracy", linewidth=2)
        plt.plot(epochs, metrics_history["test_acc"],
                 label="Test Accuracy", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Classification Accuracy: {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)

        save_path = f"plots/training_metrics/{model_name}_accuracy.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # --------------------------------------------------
    # 3. Reconstruction Loss
    # --------------------------------------------------
    if "train_recon_loss" in metrics_history and "test_recon_loss" in metrics_history:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, metrics_history["train_recon_loss"],
                 label="Train Recon Loss", linewidth=2)
        plt.plot(epochs, metrics_history["test_recon_loss"],
                 label="Test Recon Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.title(f"Reconstruction Loss: {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)

        save_path = f"plots/training_metrics/{model_name}_recon_loss.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    print("Training metric plots saved.")



def plot_pattern_comparison_bar(results_per_pattern, model_name, recon_timesteps,class_timesteps):
    """Plot bar chart comparing all patterns for a single model."""
    patterns = list(results_per_pattern.keys())
    classes = ["Square", "Random", "All-in", "All-out"]
    
    x = np.arange(len(patterns))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    plt.figure(figsize=(14, 7))
    
    for i, cls in enumerate(classes):
        cls_values = [results_per_pattern[p].get(cls, 0.0) for p in patterns]
        plt.bar(x + i * width, cls_values, width, label=cls, color=colors[i])
    
    plt.xticks(x + width * 1.5, patterns, rotation=20, ha='right')
    plt.ylabel('Max Accuracy (%)', fontsize=12)
    plt.xlabel('Test Pattern', fontsize=12)
    plt.title(f'Pattern Comparison\nModel: {model_name} (trained {trained_timesteps} timesteps)', fontsize=14)
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 100)
    plt.tight_layout()
    
    filename = f"bar_model-train_recont{recon_timesteps}_classt{class_timesteps}_{sanitize_name(model_name.replace('pc_illusiont10_recon_noise_', ''))}.png"
    filepath = os.path.join(BAR_PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Bar plot saved: {filepath}", "success")
    return filepath


def plot_grid_heatmap(gamma_values, beta_values, illusion_matrix, model_name,
                      recon_timesteps, class_timesteps):
    """Plot heatmap for grid search results."""
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        illusion_matrix,
        xticklabels=[f"{g:.2f}" for g in gamma_values],
        yticklabels=[f"{b:.2f}" for b in beta_values],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0,
        vmax=2.0,
        cbar_kws={"label": "Illusion Index"}
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Gamma", fontsize=14)
    plt.ylabel("Beta", fontsize=14)

    plt.title(
        f"Illusion Index Heatmap\nModel: {model_name} "
        f"(trained recon {recon_timesteps}, class {class_timesteps})",
        fontsize=13
    )

    plt.tight_layout()

    # --- Clean model name ---
    clean_name = sanitize_name(model_name.replace("pc_illusiont10_recon_noise_", ""))

    # --- Fixed filename f-string ---
    filename = (
        f"heatmap_model-recont{recon_timesteps}_classt{class_timesteps}_"
        f"{clean_name}.png"
    )

    filepath = os.path.join(HEATMAPS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print_status(f"Heatmap saved: {filepath}", "success")
    return filepath



def recon_pc_loss(net,dataloader,config):
    
    total_loss=[]
    test_loss=0
    for batch_idx,batch in enumerate(dataloader):
        images,labels=batch
        images,labels=images.to(config.device),labels.to(config.device)

        ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
        ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
        ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
        ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

        ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp = net.feedforward_pass_no_dense(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

        # Re-enable gradients after feedforward_pass overwrites the tensors
        # Only enable gradients for the specific tensors that need them
        ft_AB_pc_temp.requires_grad_(True)
        ft_BC_pc_temp.requires_grad_(True)
        ft_CD_pc_temp.requires_grad_(True)
        ft_DE_pc_temp.requires_grad_(True)

        final_loss=0

        for i in range(config.timesteps):
            #print("Timestep",i)
            #print("Batch Id",batch_idx)
            ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
            final_loss+=loss_of_layers

        final_loss=final_loss/config.timesteps
        total_loss.append(final_loss.item())
        # Clear batch tensors
        del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,loss_of_layers,final_loss
        torch.cuda.empty_cache()

    test_loss=np.mean(total_loss)

    return test_loss


def eval_pc_ill_accuracy(net, dataloader, config, criterion):

    total_correct = np.zeros(config.timesteps + 1)  # Initialize accuracy buffer
    total_samples = 0
    running_loss = []
    val_recon_loss = []
    
    for images, labels, _ in dataloader:

        # Move to device
        images, labels = images.to(config.device), labels.to(config.device)

        for noise in np.arange(0, 0.35, 0.05):

            images_noisy = noisy_img(images, "gauss", round(noise, 2))

            ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32)
            ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16)
            ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8)
            ft_DE_pc_temp = torch.zeros(config.batch_size, 64, 4, 4)

            ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
            ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                images_noisy,
                ft_AB_pc_temp, ft_BC_pc_temp,
                ft_CD_pc_temp, ft_DE_pc_temp
            )

            # Re-enable grad
            ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
            ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)

            # First-step prediction accuracy
            _, predicted = torch.max(output, 1)
            total_correct[0] += (predicted == labels).sum().item()

            final_loss = 0
            recon_loss = 0

            for i in range(config.timesteps):

                output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(
                    images_noisy,
                    ft_AB_pc_temp, ft_BC_pc_temp,
                    ft_CD_pc_temp, ft_DE_pc_temp,
                    ft_EF_pc_temp,
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

    # Accuracy per timestep
    accuracy = [100 * c / total_samples for c in total_correct]
    accuracy = np.array(accuracy)

    mean_acc = np.mean(accuracy)
    test_loss = np.mean(running_loss)
    test_recon_loss = np.mean(val_recon_loss)

    return mean_acc, test_loss, test_recon_loss













