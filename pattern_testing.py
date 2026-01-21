"""
Pattern Testing for Classification Models
Tests trained models with different gamma/beta patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from colorama import Fore
from utils import clear, banner
from model_tracking import get_tracker
from network import Net
from add_noise import noisy_img
import os


# Pattern definitions
PATTERNS = {
    "Uniform": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Gamma Increasing": {
        "gamma": [0.13, 0.33, 0.53, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Gamma Decreasing": {
        "gamma": [0.53, 0.33, 0.13, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Beta Increasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33]
    },
    "Beta Decreasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.53, 0.33, 0.13, 0.33]
    },
    "Beta Inc & Gamma Dec": {
        "gamma": [0.53, 0.33, 0.13, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33]
    }
}


def test_single_pattern_single_model(model_name, pattern_name, pattern_values, test_timesteps, config):
    """
    Test a single model with a single pattern
    
    Args:
        model_name: Name of the trained model
        pattern_name: Name of the pattern to test with
        pattern_values: Dict with 'gamma' and 'beta' lists
        test_timesteps: Number of timesteps to run
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
        print(f"Model {model_name} not found!")
        return None
    
    net = Net(num_classes=num_classes, input_size=input_size).to(config.device)
    
    checkpoint_path = model_info.get('checkpoint_path')
    if not checkpoint_path:
        print(f"No checkpoint for {model_name}")
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
    
    # Update config with test pattern
    config.gammaset = [pattern_values["gamma"]]
    config.betaset = [pattern_values["beta"]]
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
    
    print(f"  Testing {model_name} with {pattern_name} pattern...")
    
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


def run_pattern_testing(model_names, test_timesteps, test_patterns, config):
    """
    Test multiple models (seeds) with different patterns
    
    Args:
        model_names: List of model names (all seeds)
        test_timesteps: Number of timesteps
        test_patterns: List of pattern names to test
        config: Configuration object
    
    Returns:
        all_results: Dict[pattern_name -> seed_results]
    """
    print(f"\n{'='*60}")
    print(f"PATTERN TESTING")
    print(f"{'='*60}")
    print(f"Models: {len(model_names)} (seeds)")
    print(f"Patterns to test: {len(test_patterns)}")
    print(f"Test timesteps: {test_timesteps}")
    print(f"{'='*60}\n")
    
    # Results structure: {pattern_name: {seed: class_results}}
    all_results = {pattern_name: {} for pattern_name in test_patterns}
    
    # Test each pattern
    for pattern_name in test_patterns:
        pattern_values = PATTERNS[pattern_name]
        
        print(f"\n{Fore.CYAN}Testing with {pattern_name} pattern...{Fore.RESET}")
        print(f"Gamma: {pattern_values['gamma']}")
        print(f"Beta:  {pattern_values['beta']}\n")
        
        # Test each seed with this pattern
        for model_name in model_names:
            class_results = test_single_pattern_single_model(
                model_name, 
                pattern_name, 
                pattern_values, 
                test_timesteps, 
                config
            )
            
            if class_results:
                all_results[pattern_name][model_name] = class_results
    
    # Plot results
    plot_pattern_testing_results(all_results, model_names[0], test_timesteps)
    
    return all_results


def plot_pattern_testing_results(all_results, base_model_name, test_timesteps):
    """
    Plot pattern testing results
    One subplot per pattern, showing mean ± std across seeds
    """
    os.makedirs("plots/pattern_testing", exist_ok=True)
    
    num_patterns = len(all_results)
    
    # Create subplots: 2 columns
    ncols = 2
    nrows = (num_patterns + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    if num_patterns == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
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
    
    for idx, (pattern_name, seed_results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Aggregate across seeds for this pattern
        pattern_aggregate = {}
        
        # Get all class names from first seed
        first_seed = list(seed_results.values())[0]
        all_classes = first_seed.keys()
        
        for cls_name in all_classes:
            pattern_aggregate[cls_name] = {
                "mean": [],
                "std": []
            }
            
            # For each timestep, aggregate across seeds
            for t in range(test_timesteps + 1):
                timestep_probs = []
                
                for seed_data in seed_results.values():
                    if cls_name in seed_data and len(seed_data[cls_name]["predictions"][t]) > 0:
                        timestep_probs.append(np.mean(seed_data[cls_name]["predictions"][t]) * 100)
                
                if timestep_probs:
                    pattern_aggregate[cls_name]["mean"].append(np.mean(timestep_probs))
                    pattern_aggregate[cls_name]["std"].append(np.std(timestep_probs))
                else:
                    pattern_aggregate[cls_name]["mean"].append(0)
                    pattern_aggregate[cls_name]["std"].append(0)
        
        # Plot this pattern
        for cls_name, cls_data in pattern_aggregate.items():
            mean = np.array(cls_data["mean"])
            std = np.array(cls_data["std"])
            timesteps = np.arange(len(mean))
            
            if len(mean) == 0:
                continue
            
            ax.plot(timesteps, mean, linewidth=2, marker='o', markersize=5,
                    label=cls_name.replace('_', ' ').title(), 
                    color=colors.get(cls_name, '#95a5a6'))
            ax.fill_between(timesteps, mean - std, mean + std, 
                            alpha=0.2, color=colors.get(cls_name, '#95a5a6'))
        
        ax.set_title(f'{pattern_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Probability (%)', fontsize=10)
        ax.set_ylim([0, 100])
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(num_patterns, len(axes)):
        axes[idx].axis('off')
    
    # Main title
    fig.suptitle(f'Pattern Testing Results\n{base_model_name.rsplit("_s", 1)[0]}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filename = f"{base_model_name.rsplit('_s', 1)[0]}_pattern_testing_t{test_timesteps}.png"
    filepath = f"plots/pattern_testing/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved pattern testing plot: {filepath}")


def calculate_illusion_index(class_results):
    """
    Calculate illusion index from class results
    
    Illusion Index = max_prob(all_in) / mean(max_prob(all_out), max_prob(random))
    
    Args:
        class_results: Results dict with predictions per class
    
    Returns:
        float: Illusion index
    """
    # Get max probabilities across all timesteps
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
    
    # Calculate illusion index
    denom = (max_allout + max_random) / 2
    illusion_index = max_allin / denom if denom > 0 else 0
    
    return illusion_index


def print_pattern_testing_summary(all_results, test_timesteps):
    """
    Print summary of pattern testing results
    """
    print(f"\n{'='*60}")
    print(f"PATTERN TESTING SUMMARY")
    print(f"{'='*60}\n")
    
    for pattern_name, seed_results in all_results.items():
        print(f"{Fore.CYAN}{pattern_name}:{Fore.RESET}")
        
        illusion_indices = []
        
        for seed_name, class_results in seed_results.items():
            illusion_idx = calculate_illusion_index(class_results)
            illusion_indices.append(illusion_idx)
            print(f"  {seed_name}: Illusion Index = {illusion_idx:.3f}")
        
        if illusion_indices:
            mean_idx = np.mean(illusion_indices)
            std_idx = np.std(illusion_indices)
            print(f"  {Fore.GREEN}Mean: {mean_idx:.3f} ± {std_idx:.3f}{Fore.RESET}")
        
        print()
    
    print(f"{'='*60}\n")
