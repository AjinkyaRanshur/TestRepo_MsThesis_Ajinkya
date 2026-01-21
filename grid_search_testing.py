"""
Grid Search Testing for Classification Models
Searches over gamma/beta hyperparameter space
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from colorama import Fore
from tqdm import tqdm
from utils import clear, banner
from model_tracking import get_tracker
from network import Net
from add_noise import noisy_img
import os


def test_single_grid_point_single_model(model_name, gamma_val, beta_val, test_timesteps, config):
    """
    Test a single model with a single (gamma, beta) configuration
    
    Args:
        model_name: Name of the trained model
        gamma_val: Gamma value (will be broadcast to all 4 layers)
        beta_val: Beta value (will be broadcast to all 4 layers)
        test_timesteps: Number of timesteps
        config: Configuration object
    
    Returns:
        class_results: Dict with results per class
    """
    from main import train_test_loader
    
    tracker = get_tracker()
    model_info = tracker.get_model(model_name)
    # NEW: Determine number of classes and dataset from model config
    model_config = model_info['config']
    dataset = model_config.get('Dataset', 'custom_illusion_dataset')
    num_classes = model_config.get('last_neurons', 6)
    input_size = 128 if dataset == "custom_illusion_dataset" else 32


    if not model_info:
        return None
    
    # Load model
    net = Net(num_classes=num_classes, input_size=input_size).to(config.device)
    
    checkpoint_path = model_info.get('checkpoint_path')
    if not checkpoint_path:
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    
    net.conv1.load_state_dict(checkpoint["conv1"])
    net.conv2.load_state_dict(checkpoint["conv2"])
    net.conv3.load_state_dict(checkpoint["conv3"])
    net.conv4.load_state_dict(checkpoint["conv4"])
    net.fc1.load_state_dict(checkpoint["fc1"])
    net.fc2.load_state_dict(checkpoint["fc2"])
    net.fc3.load_state_dict(checkpoint["fc3"])
    net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
    net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
    net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
    net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])
    
    net.eval()
    
    # Update config with grid point
    config.gammaset = [[gamma_val] * 4]
    config.betaset = [[beta_val] * 4]
    config.alphaset = [[0.1, 0.1, 0.1, 0.1]]
    config.timesteps = test_timesteps
    
    # Get test dataloader
    _, _, testloader = train_test_loader(dataset, config)    

    # Get class mapping
    test_dataset = testloader.dataset
    if hasattr(test_dataset, 'dataset'):
        class_to_idx = test_dataset.dataset.class_to_idx
    else:
        class_to_idx = test_dataset.class_to_idx
    
    # Initialize results storage
    all_classes = list(class_to_idx.keys())
    class_results = {
        cls: {
            "predictions": [[] for _ in range(test_timesteps + 1)],
            "total": 0
        }
        for cls in all_classes
    }
    
    # Run testing
    for batch_data in testloader:
        images, labels, cls_names, should_see = batch_data
        images_orig = images.to(config.device)
        labels = labels.to(config.device)
        
        # Process with noise levels
        for noise in np.arange(0, 0.35, 0.05):
            images = noisy_img(images_orig.clone(), "gauss", round(noise, 2))
            
            _, _, height, width = images.shape
            batch_size = images.size(0)
            
            ft_AB_pc_temp = torch.zeros(batch_size, 6, height, width, device=config.device)
            ft_BC_pc_temp = torch.zeros(batch_size, 16, height // 2, width // 2, device=config.device)
            ft_CD_pc_temp = torch.zeros(batch_size, 32, height // 4, width // 4, device=config.device)
            ft_DE_pc_temp = torch.zeros(batch_size, 128, height // 8, width // 8, device=config.device)
            
            # Initial feedforward
            ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                    images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp
                )
            
            ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
            ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)
            
            # Get probabilities at timestep 0
            probs = F.softmax(output, dim=1).detach().cpu().numpy()
            
            # Record timestep 0
            for i, cls_name in enumerate(cls_names):
                if cls_name in ["all_in", "all_out"]:
                    perceived_class = should_see[i]
                else:
                    perceived_class = cls_name
                
                perceived_idx = class_to_idx[perceived_class]
                class_results[cls_name]["predictions"][0].append(probs[i, perceived_idx])
                
                if noise == 0.0:
                    class_results[cls_name]["total"] += 1
            
            # Run predictive coding
            for t in range(test_timesteps):
                output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                    ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(
                        images,
                        ft_AB_pc_temp,
                        ft_BC_pc_temp,
                        ft_CD_pc_temp,
                        ft_DE_pc_temp,
                        ft_EF_pc_temp,
                        config.betaset,
                        config.gammaset,
                        config.alphaset,
                        batch_size
                    )
                
                probs = F.softmax(output, dim=1).detach().cpu().numpy()
                
                for i, cls_name in enumerate(cls_names):
                    if cls_name in ["all_in", "all_out"]:
                        perceived_class = should_see[i]
                    else:
                        perceived_class = cls_name
                    
                    perceived_idx = class_to_idx[perceived_class]
                    class_results[cls_name]["predictions"][t + 1].append(probs[i, perceived_idx])
            
            del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp
            torch.cuda.empty_cache()
    
    return class_results


def calculate_illusion_index(class_results):
    """
    Calculate illusion index from class results
    
    Illusion Index = max_prob(all_in) / mean(max_prob(all_out), max_prob(random))
    """
    max_allin = 0
    max_allout = 0
    max_random = 0
    
    for cls_name, cls_data in class_results.items():
        mean_probs = [
            np.mean(p) * 100 if len(p) > 0 else 0.0
            for p in cls_data["predictions"]
        ]
        
        max_prob = max(mean_probs) if mean_probs else 0
        
        if cls_name == "all_in":
            max_allin = max_prob
        elif cls_name == "all_out":
            max_allout = max_prob
        elif cls_name == "random":
            max_random = max_prob
    
    denom = (max_allout + max_random) / 2
    illusion_index = max_allin / denom if denom > 0 else 0
    
    return illusion_index


def run_grid_search(model_names, test_timesteps, config, 
                   gamma_range=(0.13, 0.53, 0.1), beta_range=(0.13, 0.53, 0.1)):
    """
    Run grid search over gamma/beta hyperparameter space
    
    Args:
        model_names: List of model names (all seeds)
        test_timesteps: Number of timesteps
        config: Configuration object
        gamma_range: (start, stop, step) for gamma values
        beta_range: (start, stop, step) for beta values
    
    Returns:
        results_dict: Dict[(gamma, beta) -> {seed: illusion_index}]
    """
    clear()
    banner("Grid Search")
    
    GAMMA_VALUES = np.arange(*gamma_range)
    BETA_VALUES = np.arange(*beta_range)
    
    total_experiments = len(GAMMA_VALUES) * len(BETA_VALUES) * len(model_names)
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH CONFIGURATION")
    print(f"{'='*60}")
    print(f"Models: {len(model_names)} seeds")
    print(f"Gamma range: {GAMMA_VALUES[0]:.2f} - {GAMMA_VALUES[-1]:.2f} (step {gamma_range[2]})")
    print(f"Beta range: {BETA_VALUES[0]:.2f} - {BETA_VALUES[-1]:.2f} (step {beta_range[2]})")
    print(f"Grid points: {len(GAMMA_VALUES)} x {len(BETA_VALUES)} = {len(GAMMA_VALUES) * len(BETA_VALUES)}")
    print(f"Total experiments: {total_experiments}")
    print(f"Test timesteps: {test_timesteps}")
    print(f"{'='*60}\n")
    
    # Results structure: {(gamma, beta): {seed_name: illusion_index}}
    results_dict = {}
    
    # Run grid search
    with tqdm(total=total_experiments, desc="Grid Search", unit="exp") as pbar:
        for gamma in GAMMA_VALUES:
            for beta in BETA_VALUES:
                grid_point = (round(gamma, 2), round(beta, 2))
                results_dict[grid_point] = {}
                
                pbar.set_postfix_str(f"γ={gamma:.2f}, β={beta:.2f}")
                
                # Test each seed at this grid point
                for model_name in model_names:
                    try:
                        class_results = test_single_grid_point_single_model(
                            model_name, gamma, beta, test_timesteps, config
                        )
                        
                        if class_results:
                            illusion_idx = calculate_illusion_index(class_results)
                            results_dict[grid_point][model_name] = illusion_idx
                        else:
                            results_dict[grid_point][model_name] = 0.0
                    
                    except Exception as e:
                        print(f"\n{Fore.RED}Error at γ={gamma:.2f}, β={beta:.2f}, model={model_name}: {e}{Fore.RESET}")
                        results_dict[grid_point][model_name] = 0.0
                    
                    pbar.update(1)
    
    # Plot results
    plot_grid_search_heatmap(results_dict, GAMMA_VALUES, BETA_VALUES, 
                             model_names[0], test_timesteps)
    
    return results_dict


def plot_grid_search_heatmap(results_dict, gamma_values, beta_values, 
                             base_model_name, test_timesteps):
    """
    Plot grid search results as heatmap
    Shows mean illusion index across seeds with std as annotation
    """
    os.makedirs("plots/grid_search", exist_ok=True)
    
    # Build matrices for plotting
    mean_matrix = np.zeros((len(beta_values), len(gamma_values)))
    std_matrix = np.zeros((len(beta_values), len(gamma_values)))
    
    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            grid_point = (round(gamma, 2), round(beta, 2))
            
            if grid_point in results_dict:
                indices = list(results_dict[grid_point].values())
                if indices:
                    mean_matrix[i, j] = np.mean(indices)
                    std_matrix[i, j] = np.std(indices)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(mean_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=0, vmax=2.0, origin='lower')
    
    # Set ticks
    ax.set_xticks(np.arange(len(gamma_values)))
    ax.set_yticks(np.arange(len(beta_values)))
    ax.set_xticklabels([f'{g:.2f}' for g in gamma_values])
    ax.set_yticklabels([f'{b:.2f}' for b in beta_values])
    
    # Labels
    ax.set_xlabel('Gamma (Forward)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Beta (Backward)', fontsize=13, fontweight='bold')
    ax.set_title(f'Grid Search: Illusion Index Heatmap\n{base_model_name.rsplit("_s", 1)[0]}', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Illusion Index (mean across seeds)', fontsize=11)
    
    # Annotate cells with mean ± std
    for i in range(len(beta_values)):
        for j in range(len(gamma_values)):
            mean_val = mean_matrix[i, j]
            std_val = std_matrix[i, j]
            
            text_color = 'white' if mean_val > 1.0 else 'black'
            
            ax.text(j, i, f'{mean_val:.2f}\n±{std_val:.2f}',
                   ha="center", va="center", color=text_color, fontsize=8)
    
    # Add grid
    ax.set_xticks(np.arange(len(gamma_values)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(beta_values)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save
    filename = f"{base_model_name.rsplit('_s', 1)[0]}_grid_search_t{test_timesteps}.png"
    filepath = f"plots/grid_search/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved grid search heatmap: {filepath}")


def print_grid_search_summary(results_dict, gamma_values, beta_values):
    """
    Print summary of grid search results
    """
    print(f"\n{'='*60}")
    print(f"GRID SEARCH SUMMARY")
    print(f"{'='*60}\n")
    
    # Find best configuration
    best_illusion_idx = 0
    best_gamma = 0
    best_beta = 0
    
    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            grid_point = (round(gamma, 2), round(beta, 2))
            
            if grid_point in results_dict:
                indices = list(results_dict[grid_point].values())
                if indices:
                    mean_idx = np.mean(indices)
                    
                    if mean_idx > best_illusion_idx:
                        best_illusion_idx = mean_idx
                        best_gamma = gamma
                        best_beta = beta
    
    print(f"{Fore.GREEN}Best Configuration:{Fore.RESET}")
    print(f"  Gamma: {best_gamma:.2f}")
    print(f"  Beta:  {best_beta:.2f}")
    print(f"  Illusion Index: {best_illusion_idx:.3f}")
    
    # Show top 5 configurations
    print(f"\n{Fore.CYAN}Top 5 Configurations:{Fore.RESET}")
    
    all_configs = []
    for grid_point, seed_results in results_dict.items():
        if seed_results:
            mean_idx = np.mean(list(seed_results.values()))
            std_idx = np.std(list(seed_results.values()))
            all_configs.append((grid_point, mean_idx, std_idx))
    
    all_configs.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (grid_point, mean_idx, std_idx) in enumerate(all_configs[:5], 1):
        gamma, beta = grid_point
        print(f"  {rank}. γ={gamma:.2f}, β={beta:.2f}: {mean_idx:.3f} ± {std_idx:.3f}")
    
    print(f"\n{'='*60}\n")
