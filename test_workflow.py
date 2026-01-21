"""
Unified testing workflow for classification models
Supports: trajectory testing, pattern testing, and grid search
"""

import numpy as np
import matplotlib.pyplot as plt
from model_tracking import get_tracker
from menu_options import select_test_models, select_test_timesteps, testing_type_menu
from utils import clear, banner
from colorama import Fore
import torch
import torch.nn.functional as F
from add_noise import noisy_img
from network import Net


def run_trajectory_test(model_names, test_timesteps, config):
    """
    Test model trajectory over timesteps
    Combines results from multiple seeds with error bars
    """
    print(f"\n{'='*60}")
    print(f"TRAJECTORY TESTING")
    print(f"{'='*60}")
    print(f"Models: {len(model_names)}")
    print(f"Test timesteps: {test_timesteps}")
    print(f"{'='*60}\n")
    
    tracker = get_tracker()
    
    # Aggregate results across seeds
    all_class_results = {}
    
    for model_name in model_names:
        model_info = tracker.get_model(model_name)
        if not model_info:
            continue
        
        # Load model and run test
        class_results = run_single_trajectory_test(model_name, test_timesteps, config)
        
        # Aggregate results
        for cls_name, cls_data in class_results.items():
            if cls_name not in all_class_results:
                all_class_results[cls_name] = {
                    "predictions_all_seeds": [],
                    "total": cls_data["total"]
                }
            
            all_class_results[cls_name]["predictions_all_seeds"].append(
                cls_data["predictions"]
            )
    
    # Plot aggregated trajectory with error bars
    plot_trajectory_with_seeds(all_class_results, model_names[0], test_timesteps)
    
    return all_class_results


def run_single_trajectory_test(model_name, test_timesteps, config):
    """
    Run trajectory test on a single model
    Returns class_results dict
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


def plot_trajectory_with_seeds(all_class_results, base_model_name, test_timesteps):
    """
    Plot trajectory with error bars across seeds
    """
    import os
    os.makedirs("plots/test_trajectories", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
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
    
    for cls_name, cls_data in all_class_results.items():
        if cls_data["total"] == 0:
            continue
        
        # Aggregate across seeds
        predictions_all_seeds = cls_data["predictions_all_seeds"]
        
        # Calculate mean and std across seeds
        mean_trajectory = []
        std_trajectory = []
        
        for t in range(test_timesteps + 1):
            # Get all predictions at this timestep across all seeds
            timestep_data = []
            for seed_preds in predictions_all_seeds:
                if len(seed_preds[t]) > 0:
                    timestep_data.append(np.mean(seed_preds[t]) * 100)
            
            if timestep_data:
                mean_trajectory.append(np.mean(timestep_data))
                std_trajectory.append(np.std(timestep_data))
            else:
                mean_trajectory.append(0)
                std_trajectory.append(0)
        
        mean_trajectory = np.array(mean_trajectory)
        std_trajectory = np.array(std_trajectory)
        
        timesteps = np.arange(len(mean_trajectory))
        
        # Plot with error bars
        ax.plot(timesteps, mean_trajectory, linewidth=2.5, marker='o', markersize=6,
                label=cls_name.replace('_', ' ').title(), color=colors.get(cls_name, '#95a5a6'))
        ax.fill_between(timesteps, mean_trajectory - std_trajectory, 
                        mean_trajectory + std_trajectory, alpha=0.2, color=colors.get(cls_name, '#95a5a6'))
    
    ax.set_xlabel('Timestep', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability of Correct Class (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'PC Dynamics Trajectory\n{base_model_name.rsplit("_s", 1)[0]}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save with unique filename
    filename = f"{base_model_name.rsplit('_s', 1)[0]}_trajectory_t{test_timesteps}.png"
    filepath = f"plots/test_trajectories/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved trajectory plot: {filepath}")


def run_pattern_testing(model_names, test_timesteps, config):
    """
    Test models with different patterns
    """
    from pattern_testing import run_pattern_testing as pattern_test, print_pattern_testing_summary, PATTERNS
    
    # Let user select patterns to test
    clear()
    banner("Pattern Selection")
    
    print(Fore.YELLOW + "Available patterns:\n")
    pattern_list = list(PATTERNS.keys())
    for i, p in enumerate(pattern_list, 1):
        print(f"{Fore.GREEN}  {i}. {p}")
    
    print(Fore.YELLOW + "\nSelect patterns to test:")
    print("  - Enter pattern numbers (comma-separated, e.g., 1,2,3)")
    print("  - Or enter 'all' for all patterns\n")
    
    choice = input(Fore.WHITE + "Your choice: ").strip()
    
    if choice.lower() == 'all':
        test_patterns = pattern_list
    else:
        try:
            indices = [int(x.strip())-1 for x in choice.split(',')]
            test_patterns = [pattern_list[i] for i in indices if 0 <= i < len(pattern_list)]
        except (ValueError, IndexError):
            print(Fore.RED + "Invalid selection!")
            input("Press ENTER...")
            return
    
    # Run pattern testing
    all_results = pattern_test(model_names, test_timesteps, test_patterns, config)
    
    # Print summary
    print_pattern_testing_summary(all_results, test_timesteps)
    
    input("\nPress ENTER to continue...")


def run_grid_search(model_names, test_timesteps, config):
    """
    Grid search over hyperparameters
    """
    from grid_search_testing import run_grid_search as grid_search, print_grid_search_summary
    
    # Get grid search parameters
    clear()
    banner("Grid Search Setup")
    
    print(Fore.YELLOW + "Configure grid search parameters:\n")
    
    # Gamma range
    print("Gamma range:")
    gamma_start = float(input("  Start (default 0.13): ").strip() or "0.13")
    gamma_stop = float(input("  Stop (default 0.53): ").strip() or "0.53")
    gamma_step = float(input("  Step (default 0.1): ").strip() or "0.1")
    
    # Beta range
    print("\nBeta range:")
    beta_start = float(input("  Start (default 0.13): ").strip() or "0.13")
    beta_stop = float(input("  Stop (default 0.53): ").strip() or "0.53")
    beta_step = float(input("  Step (default 0.1): ").strip() or "0.1")
    
    gamma_range = (gamma_start, gamma_stop, gamma_step)
    beta_range = (beta_start, beta_stop, beta_step)
    
    # Run grid search
    results = grid_search(model_names, test_timesteps, config, gamma_range, beta_range)
    
    # Print summary
    import numpy as np
    gamma_values = np.arange(*gamma_range)
    beta_values = np.arange(*beta_range)
    print_grid_search_summary(results, gamma_values, beta_values)
    
    input("\nPress ENTER to continue...")


def run_testing_workflow():
    """
    Main testing workflow
    """
    # Step 1: Select test type
    test_type = testing_type_menu()
    
    if test_type == "0":
        return
    
    # Step 2: Select models
    model_names = select_test_models()
    
    if not model_names:
        return
    
    # Step 3: Get test timesteps
    test_timesteps = select_test_timesteps()
    
    # Step 4: Load config for the model
    tracker = get_tracker()
    first_model = tracker.get_model(model_names[0])
    
    if not first_model:
        print("Model not found!")
        return
    
    # Create a minimal config object
    class TestConfig:
        def __init__(self, model_config):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.batch_size = 40
            self.timesteps = test_timesteps
            self.seed = model_config.get('seed', 42)
            
            # Pattern values (will be overridden for pattern testing)
            self.gammaset = [[0.33, 0.33, 0.33, 0.33]]
            self.betaset = [[0.33, 0.33, 0.33, 0.33]]
            self.alphaset = [[0.1, 0.1, 0.1, 0.1]]
            
            self.classification_datasetpath = model_config.get('Dataset', 'custom_illusion_dataset')
            self.recon_datasetpath = model_config.get('Dataset', 'custom_illusion_dataset')
    
    config = TestConfig(first_model['config'])
    
    # Step 5: Run appropriate test
    if test_type == "1":
        # Trajectory testing
        run_trajectory_test(model_names, test_timesteps, config)
    elif test_type == "2":
        # Pattern testing
        run_pattern_testing(model_names, test_timesteps, config)
    elif test_type == "3":
        # Grid search
        run_grid_search(model_names, test_timesteps, config)
    else:
        print(Fore.RED + "Invalid test type!")
    
    input("\nPress ENTER to continue...")
