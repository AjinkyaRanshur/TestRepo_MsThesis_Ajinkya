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



def plot_trajectory(results, pattern_name, model_name,recon_timesteps,class_timesteps, test_timesteps):
    """Plot accuracy trajectory over timesteps."""
    accuracy_data = {}
    
    for cls_name in ["Square", "Random", "All-in", "All-out"]:
        if cls_name in results:
            mean_probs = [np.mean(p) * 100 for p in results[cls_name]["predictions"]]
            timesteps_range = list(range(len(mean_probs)))
            accuracy_data[cls_name] = {'timesteps': timesteps_range, 'values': mean_probs}
    
    plt.figure(figsize=(12, 6))
    colors = {'Square': '#2ecc71', 'Random': '#3498db', 'All-in': '#e74c3c', 'All-out': '#9b59b6'}
    
    for cls_name, data in accuracy_data.items():
        plt.plot(data['timesteps'], data['values'], linestyle='-', linewidth=2, markersize=6,
                label=cls_name, color=colors.get(cls_name, '#95a5a6'))
    
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Probability of Being A Shape (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.title(f'PC Dynamics Trajectory\nModel: {model_name} | Test Pattern: {pattern_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # Save with organized naming
    filename = f"traj_model-train_recont{recon_timesteps}_classt{class_timesteps}_{sanitize_name(pattern_name)}_test-t{test_timesteps}.png"
    filepath = os.path.join(TRAJECTORIES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Trajectory plot saved: {filepath}", "success")
    return filepath


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













