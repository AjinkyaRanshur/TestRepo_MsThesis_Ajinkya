import os
import pyfiglet
from colorama import Fore, Style, init
import sys
from main import load_config, main
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

# init colorama for Windows
init(autoreset=True)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG_FILE = "configs/configtest.py"
CONFIG_MODULE = "configtest"

# Directory structure for organized outputs
BASE_RESULTS_DIR = "result_folder"
TRAJECTORIES_DIR = os.path.join(BASE_RESULTS_DIR, "trajectories")
BAR_PLOTS_DIR = os.path.join(BASE_RESULTS_DIR, "bar_plots")
HEATMAPS_DIR = os.path.join(BASE_RESULTS_DIR, "heatmaps")
SUMMARIES_DIR = os.path.join(BASE_RESULTS_DIR, "summaries")

# Create directories
for d in [BASE_RESULTS_DIR, TRAJECTORIES_DIR, BAR_PLOTS_DIR, HEATMAPS_DIR, SUMMARIES_DIR]:
    os.makedirs(d, exist_ok=True)

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

# Available trained timesteps
AVAILABLE_MODELS = {
    "recon_t1_class_t1": {"recon": 1, "class": 1},
    "recon_t1_class_t10": {"recon": 1, "class": 10},
    "recon_t10_class_t1": {"recon": 10, "class": 1},
    "recon_t10_class_t10": {"recon": 10, "class": 10},
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def banner(text):
    print(Fore.CYAN + pyfiglet.figlet_format(text, font="ogre"))

def sanitize_name(name):
    """Convert pattern name to valid filename."""
    return name.lower().replace(' ', '_').replace('.', '').replace('&', 'and')

# REPLACE get_model_name function:
def get_model_name(pattern_name, recon_timesteps, class_timesteps=None):
    """
    Generate model name based on pattern and training timesteps.
    
    Args:
        pattern_name: Name of the pattern
        recon_timesteps: Timesteps used for reconstruction training
        class_timesteps: Timesteps used for classification training (optional)
    
    Returns:
        Model name in format: pc_recon_t{R}_class_t{C}_{pattern}
        or pc_recon_t{R}_{pattern} if class_timesteps is None
    """
    sanitized = sanitize_name(pattern_name)
    
    if class_timesteps is not None:
        return f"pc_recon_t{recon_timesteps}_class_t{class_timesteps}_{sanitized}"
    else:
        # For reconstruction-only models
        return f"pc_recon_t{recon_timesteps}_{sanitized}"

def progress_bar(current, total, prefix="Progress", length=40):
    """Display a progress bar."""
    percent = current / total
    filled = int(length * percent)
    bar = Fore.GREEN + "█" * filled + Fore.WHITE + "░" * (length - filled)
    print(f"\r{Fore.YELLOW}{prefix}: {bar} {Fore.CYAN}{current}/{total} ({percent*100:.1f}%)", end="", flush=True)

def spinner(duration=2, message="Processing"):
    """Display a spinner animation."""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        print(f"\r{Fore.YELLOW}{message} {Fore.CYAN}{chars[i % len(chars)]}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print("\r" + " " * (len(message) + 5), end="\r")

def print_box(title, content_lines, color=Fore.CYAN):
    """Print content in a styled box."""
    max_len = max(len(title), max(len(line) for line in content_lines)) + 4
    print(color + "╔" + "═" * max_len + "╗")
    print(color + "║ " + Fore.YELLOW + title.center(max_len - 2) + color + " ║")
    print(color + "╠" + "═" * max_len + "╣")
    for line in content_lines:
        print(color + "║ " + Fore.WHITE + line.ljust(max_len - 2) + color + " ║")
    print(color + "╚" + "═" * max_len + "╝")

def print_status(message, status="info"):
    """Print a status message with icon."""
    icons = {"info": "ℹ", "success": "✓", "error": "✗", "warning": "⚠", "running": "▶"}
    colors = {"info": Fore.BLUE, "success": Fore.GREEN, "error": Fore.RED, "warning": Fore.YELLOW, "running": Fore.MAGENTA}
    print(f"{colors.get(status, Fore.WHITE)} {icons.get(status, '•')} {message}")

# ============================================================
# CONFIG AND MODEL FUNCTIONS
# ============================================================
def run_and_analyze():
    """Load config and run training/analysis."""
    configs_path = os.path.abspath("configs")
    if configs_path not in sys.path:
        sys.path.insert(0, configs_path)
    
    if CONFIG_MODULE in sys.modules:
        del sys.modules[CONFIG_MODULE]
    
    config = load_config(CONFIG_MODULE)
    results = main(config)
    return results

def update_config(gamma_pattern, beta_pattern, pattern_name, model_name, timesteps, iterations=4):
    """Update the config file with new parameters."""
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()
    
    with open(CONFIG_FILE, "w") as f:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("gammaset"):
                f.write(f"gammaset = [[{', '.join(f'{g:.2f}' for g in gamma_pattern)}]]  # pattern: {pattern_name}\n")
            elif stripped.startswith("betaset"):
                f.write(f"betaset = [[{', '.join(f'{b:.2f}' for b in beta_pattern)}]]  # pattern: {pattern_name}\n")
            elif stripped.startswith("model_name"):
                f.write(f'model_name = "{model_name}"\n')
            elif stripped.startswith("timesteps"):
                f.write(f"timesteps = {timesteps}\n")
            elif stripped.startswith("iterations"):
                f.write(f"iterations = {iterations}\n")
            elif stripped.startswith("experiment_name"):
                f.write(f'experiment_name = "Testing {model_name} with {pattern_name} pattern at {timesteps} timesteps"\n')
            else:
                f.write(line)

# ============================================================
# MENU FUNCTIONS
# ============================================================
def main_menu():
    clear()
    banner("Pred-Net Analyzer")
    print(Fore.YELLOW + "Select an option:\n")
    print(Fore.GREEN + " [1] Train the Model")
    print(Fore.GREEN + " [2] Test the Model")
    print(Fore.RED   + " [0] Exit\n")
    return input(Fore.WHITE + "Enter choice: ")

def train_menu():
    clear()
    banner("Training")
    print(Fore.YELLOW + "Training Options:\n")
    print(Fore.GREEN + " [1] Reconstruction Training")
    print(Fore.GREEN + " [2] Classification Training")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def classification_train_menu():
    clear()
    banner("Classification Training")
    print(Fore.YELLOW + "Training Options:\n")
    print(Fore.GREEN + " [1] Training On Specific Pattern")
    print(Fore.GREEN + " [2] Training With All Patterns")
    print(Fore.GREEN + " [3] Training With Learnable Hyperparameters")
    print(Fore.GREEN + " [4] Training with Weighting Over Timesteps")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def test_menu():
    clear()
    banner("Testing")
    print(Fore.YELLOW + "Testing Options:\n")
    print(Fore.GREEN + " [1] Reconstruction Models")
    print(Fore.GREEN + " [2] Classification Models")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def classification_test_menu():
    clear()
    banner("Classification Testing")
    print(Fore.YELLOW + "Testing Options:\n")
    print(Fore.GREEN + " [1] Test Single Model (One Pattern)")
    print(Fore.GREEN + " [2] Test Single Model (All Patterns)")
    print(Fore.GREEN + " [3] Test All Models (All Patterns)")
    print(Fore.GREEN + " [4] Grid Search on Hyperparameters")
    print(Fore.RED   + " [0] Back\n")
    return input(Fore.WHITE + "Enter choice: ")

def select_trained_timesteps():
    """Let user select which trained model timesteps to use."""
    clear()
    banner("Select Model")
    print(Fore.YELLOW + "Select the model based on training timesteps:\n")
    
    model_names = list(AVAILABLE_MODELS.keys())
    for idx, model_key in enumerate(model_names, 1):
        recon_t = AVAILABLE_MODELS[model_key]["recon"]
        class_t = AVAILABLE_MODELS[model_key]["class"]
        print(Fore.GREEN + f" [{idx}] Recon: {recon_t}t, Classification: {class_t}t")
    print(Fore.RED + " [0] Back\n")
    
    try:
        choice = int(input(Fore.WHITE + "Enter choice: "))
        if choice == 0:
            return None
        if 1 <= choice <= len(model_names):
            return AVAILABLE_MODELS[model_names[choice - 1]]
    except ValueError:
        pass
    
    print(Fore.RED + "Invalid choice!")
    input("Press ENTER...")
    return None

def select_pattern():
    """Let user select a specific pattern."""
    clear()
    banner("Select Pattern")
    print(Fore.YELLOW + "Available patterns:\n")
    
    pattern_names = list(PATTERNS.keys())
    for idx, name in enumerate(pattern_names, 1):
        gamma = PATTERNS[name]["gamma"]
        beta = PATTERNS[name]["beta"]
        print(Fore.GREEN + f" [{idx}] {name}")
        print(Fore.WHITE + f"      Gamma: {gamma}")
        print(Fore.WHITE + f"      Beta:  {beta}\n")
    
    print(Fore.RED + " [0] Back\n")
    
    try:
        choice = int(input(Fore.WHITE + "Enter choice: "))
        if choice == 0:
            return None, None
        if 1 <= choice <= len(pattern_names):
            name = pattern_names[choice - 1]
            return name, PATTERNS[name]
    except ValueError:
        pass
    
    print(Fore.RED + "Invalid choice!")
    input("Press ENTER...")
    return None, None

def get_test_timesteps():
    """Get testing timesteps from user."""
    clear()
    banner("Test Parameters")
    print(Fore.YELLOW + "Configure testing parameters:\n")
    
    try:
        timesteps = int(input(Fore.WHITE + "Enter number of PC timesteps for testing (default 10): ") or "10")
    except ValueError:
        print(Fore.RED + "Invalid input, using default (10)")
        timesteps = 10
    
    return timesteps

def get_training_params():
    """Get timesteps and iterations from user."""
    clear()
    banner("Training Parameters")
    print(Fore.YELLOW + "Configure training parameters:\n")
    
    try:
        timesteps = int(input(Fore.WHITE + "Enter number of timesteps (default 10): ") or "10")
        iterations = int(input(Fore.WHITE + "Enter number of iterations (default 4): ") or "4")
    except ValueError:
        print(Fore.RED + "Invalid input, using defaults")
        timesteps, iterations = 10, 4
    
    return timesteps, iterations

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def reconstruction_training():

	return results


def train_single_pattern(pattern_name, pattern_values, recon_timesteps,class_timesteps,iterations):
    """Train a single pattern."""
    gamma_pattern = pattern_values["gamma"]
    beta_pattern = pattern_values["beta"]
    model_name = get_model_name(pattern_name,recon_timesteps,class_timesteps)
    
    print_box(" Classification Training Configuration", [
        f"Pattern: {pattern_name}",
        f"Model: {model_name}",
        f"Gamma: {gamma_pattern}",
        f"Beta: {beta_pattern}",
        f"Timesteps: {timesteps}",
        f"Iterations: {iterations}"
	f"Recon Timesteps:{recon_timesteps}",
	f"Class Timesteps:{class_timesteps}"
    ])
    
    try:
        print_status("Updating configuration...", "running")
        update_config(gamma_pattern, beta_pattern, pattern_name, model_name, train_timesteps, iterations)
        
        print_status("Starting training...", "running")
        results = run_and_analyze()
        
        print_status(f"Training completed for {pattern_name}", "success")
        return results
    except Exception as e:
        print_status(f"Training failed: {str(e)}", "error")
        return None

def train_all_patterns(timesteps, iterations):
    """Train all patterns sequentially."""
    clear()
    banner("Training All Patterns")
    
    total = len(PATTERNS)
    results = {}

    with tqdm(total=len(PATTERNS),desc="Training Patterns",unit="pattern",bar_format='{l_bar}{bar:30}{r_bar}' ) as pbar:
    
    	for idx, (pattern_name, pattern_values) in enumerate(PATTERNS.items(), 1):
		pbar.set_postfix_str(f"Current: {pattern_name}")
        	result = train_single_pattern(pattern_name, pattern_values,recon_timesteps,class_timesteps)
        	results[pattern_name] = result
		pbar.update(1)
    
    # Summary
    print("\n")
    summary_lines = []
    for name, result in results.items():
        status = "✓ Success" if result is not None else "✗ Failed"
        summary_lines.append(f"{name}: {status}")
    
    print_box("TRAINING SUMMARY", summary_lines, Fore.GREEN)
    return results

# ============================================================
# TESTING FUNCTIONS
# ============================================================
def test_single_pattern(pattern_name, pattern_values, model_name,recon_timesteps, class_timesteps,test_timesteps):
    """Test a single pattern on a specific model."""
    gamma_pattern = pattern_values["gamma"]
    beta_pattern = pattern_values["beta"]
    
    print_box("Testing Configuration", [
        f"Model: {model_name}",
        f"Trained with: recon_t{recon_timesteps}, class_t{class_timesteps}",
        f"Testing pattern: {pattern_name}",
        f"Test timesteps: {test_timesteps}",
        f"Gamma: {gamma_pattern}",
        f"Beta: {beta_pattern}"
    ])
    
    try:
        print_status("Updating configuration...", "running")
        update_config(gamma_pattern, beta_pattern, pattern_name, model_name, test_timesteps, iterations=1)
        
        print_status("Running evaluation...", "running")
        spinner(1, "Loading model")
        results = run_and_analyze()
        
        print_status(f"Testing completed", "success")
        return results
    except Exception as e:
        print_status(f"Testing failed: {str(e)}", "error")
        return None

def test_model_all_patterns(trained_pattern_name,model_config, test_timesteps):
    """Test a model against all patterns."""
    clear()
    banner("Testing All Patterns")

    recon_t = model_config["recon"]
    class_t = model_config["class"]
    model_name = get_model_name(trained_pattern_name, recon_t, class_t)

    all_results = {}
    
    print_status(f"Testing model: {model_name}", "info")
    print_status(f"Against all {total} patterns\n", "info")

    with tqdm(total=len(PATTERNS),desc="Testing Patterns",unit="pattern",bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
    
    	for pattern_name, pattern_values in PATTERNS.items():
        	pbar.set_postfix_str(f"Current: {pattern_name}")
            
            	results = test_single_pattern(pattern_name, pattern_values, model_name, 
                                         recon_t, class_t, test_timesteps)
            	if results:
                	# Extract max accuracy per class
                	class_results = {}
                	for cls_name in ["Square", "Random", "All-in", "All-out"]:
                    		if cls_name in results:
                        		mean_probs = [np.mean(p) * 100 for p in results[cls_name]["predictions"]]
                        		class_results[cls_name] = max(mean_probs)
                	all_results[pattern_name] = class_results
                
                	# Plot trajectory for this pattern
                	plot_trajectory(results, pattern_name, model_name, recon_t,class_t test_timesteps)
            
            	pbar.update(1)
    
    print("\n")
    
    # Plot bar chart summary
    if all_results:
        plot_pattern_comparison_bar(all_results, model_name, recon_t)
	
    return all_results

def test_all_models_all_patterns(test_timesteps):
    """Test all trained models against all patterns."""
    clear()
    banner("Full Model Evaluation")
    
    all_model_results = {}

    model_names = list(AVAILABLE_MODELS.keys())
    
    for idx,model_key in enumerate(model_names,1):
	recon_t = AVAILABLE_MODELS[model_key]["recon"]
        class_t = AVAILABLE_MODELS[model_key]["class"]
        print(f"\n{Fore.YELLOW}{'='*60}")
        print(f"{Fore.YELLOW}Models trained with Reconstruction: {recon_t} timesteps and Classfication : {class_t} timesteps")
        print(f"{Fore.YELLOW}{'='*60}")
        
        for pattern_name in PATTERNS.keys():
            model_name = get_model_name(pattern_name,recon_timesteps,class_timesteps)
            print(f"\n{Fore.MAGENTA}Testing model: {model_name}")
            
            model_results = {}
            for test_pattern_name, test_pattern_values in PATTERNS.items():
                results = test_single_pattern(
                    test_pattern_name, test_pattern_values,
                    model_name, recon_timesteps,class_timesteps, test_timesteps
                )
                if results:
                    class_results = {}
                    for cls_name in ["Square", "Random", "All-in", "All-out"]:
                        if cls_name in results:
                            mean_probs = [np.mean(p) * 100 for p in results[cls_name]["predictions"]]
                            class_results[cls_name] = max(mean_probs)
                    model_results[test_pattern_name] = class_results
            
            all_model_results[f"{pattern_name}_t{trained_ts}"] = model_results
    
    return all_model_results

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
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
    plt.ylabel('Probability of Being Square (%)', fontsize=12)
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

def plot_grid_heatmap(gamma_values, beta_values, illusion_matrix, model_name, recon_timesteps,class_timesteps):
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
        vmax=max(1.6, np.max(illusion_matrix)),
        cbar_kws={"label": "Illusion Index"}
    )
    
    plt.gca().invert_yaxis()
    plt.xlabel("Gamma", fontsize=14)
    plt.ylabel("Beta", fontsize=14)
    plt.title(f"Illusion Index Heatmap\nModel: {model_name} (trained {trained_timesteps} timesteps)", fontsize=13)
    plt.tight_layout()
    
    filename = f"heatmap_model-t{trained_recont{recon_timesteps}_classt{class_timesteps}_{sanitize_name(model_name.replace('pc_illusiont10_recon_noise_', ''))}.png"
    filepath = os.path.join(HEATMAPS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Heatmap saved: {filepath}", "success")
    return filepath

# ============================================================
# GRID SEARCH FUNCTION
# ============================================================
def run_grid_search(trained_pattern_name, model_config, test_timesteps):
    """Run grid search over gamma and beta values."""
    clear()
    banner("Grid Search")

    recon_t=model_config["recon"]
    class_t=model_config["class"]
    
    model_name = get_model_name(trained_pattern_name, recon_timesteps,class_timesteps)
    
    GAMMA_VALUES = np.arange(0.13, 0.44, 0.1)
    BETA_VALUES = np.arange(0.13, 0.44, 0.1)
    
    results_dict = {}
    total = len(GAMMA_VALUES) * len(BETA_VALUES)
    count = 0
    
    print_box("Grid Search Configuration", [
        f"Model: {model_name}",
        f"Gamma range: {GAMMA_VALUES[0]:.2f} - {GAMMA_VALUES[-1]:.2f}",
        f"Beta range: {BETA_VALUES[0]:.2f} - {BETA_VALUES[-1]:.2f}",
        f"Total experiments: {total}"
    ])
    
    with tqdm(total=total,desc="Grid Search",unit="config",bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
    	for gamma in GAMMA_VALUES:
        	for beta in BETA_VALUES:
            		pbar.set_postfix_str(f"γ={gamma:.2f}, β={beta:.2f}")
            		gamma_pattern = [gamma, gamma, gamma, gamma]
            		beta_pattern = [beta, beta, beta, beta]
            
            		try:
                		update_config(gamma_pattern, beta_pattern, f"Grid_g{gamma:.2f}_b{beta:.2f}", 
                            		model_name, test_timesteps, iterations=1)
                		results = run_and_analyze()
                
                		if results:
                    			class_results = {}
                    			for cls_name in ["Square", "Random", "All-in", "All-out"]:
                        			if cls_name in results:
                            				mean_probs = [np.mean(p) * 100 for p in results[cls_name]["predictions"]]
                            				class_results[cls_name] = max(mean_probs)
                    			results_dict[(gamma, beta)] = class_results
            		except Exception as e:
                	print(f"\n{Fore.RED}Error at gamma={gamma:.2f}, beta={beta:.2f}: {e}")
                	results_dict[(gamma, beta)] = {}
			
			pbar.update(1)

    
    # Build illusion matrix
    illusion_matrix = np.zeros((len(BETA_VALUES), len(GAMMA_VALUES)))
    
    for i, beta in enumerate(BETA_VALUES):
        for j, gamma in enumerate(GAMMA_VALUES):
            res = results_dict.get((gamma, beta), {})
            max_allin = res.get("All-in", 0)
            max_allout = res.get("All-out", 0)
            max_random = res.get("Random", 0)
            
            denom = (max_allout + max_random) / 2
            illusion_index = max_allin / denom if denom > 0 else 0
            illusion_matrix[i, j] = illusion_index
    
    print("\n")
    plot_grid_heatmap(GAMMA_VALUES, BETA_VALUES, illusion_matrix, model_name, recon_timesteps,class_timesteps)
    
    return results_dict, illusion_matrix

# ============================================================
# MAIN RUN FUNCTION
# ============================================================
def run():
    while True:
        choice = main_menu()

        if choice == "1":
            # TRAINING MENU
            while True:
                t = train_menu()
                if t == "1":
                    clear()
                    banner("Reconstruction Training")
                    print(Fore.YELLOW + "Coming soon...")
                    input("\nPress ENTER to go back...")
                    
                elif t == "2":
                    while True:
                        cl = classification_train_menu()
                        
                        if cl == "1":
                            pattern_name, pattern_values = select_pattern()
                            if pattern_name:
                                timesteps, iterations = get_training_params()
                                train_single_pattern(pattern_name, pattern_values, timesteps, iterations)
                                input("\nPress ENTER to continue...")
                                
                        elif cl == "2":
                            timesteps, iterations = get_training_params()
                            confirm = input(Fore.YELLOW + f"\nTrain all {len(PATTERNS)} patterns? (y/n): ").lower()
                            if confirm == 'y':
                                train_all_patterns(timesteps, iterations)
                            input("\nPress ENTER to continue...")
                            
                        elif cl == "3":
                            clear()
                            banner("Learnable Hyperparams")
                            print(Fore.YELLOW + "Coming soon...")
                            input("\nPress ENTER to go back...")
                            
                        elif cl == "4":
                            clear()
                            banner("Timestep Weighting")
                            print(Fore.YELLOW + "Coming soon...")
                            input("\nPress ENTER to go back...")
                            
                        elif cl == "0":
                            break
                        else:
                            print(Fore.RED + "Invalid option!")
                            input("Press ENTER...")
                            
                elif t == "0":
                    break
                else:
                    print(Fore.RED + "Invalid option!")
                    input("Press ENTER...")

        elif choice == "2":
            # TESTING MENU
            while True:
                t = test_menu()
                if t == "1":
                    clear()
                    banner("Reconstruction Models")
                    print(Fore.YELLOW + "Coming soon...")
                    input("\nPress ENTER to go back...")
                    
                elif t == "2":
                    while True:
                        cl = classification_test_menu()
                        
                        if cl == "1":
                            # Test single model with single pattern
                            model_config = select_trained_timesteps()
                            if model_config is None:
                                continue
                            
                            trained_pattern, trained_values = select_pattern()
                            if trained_pattern is None:
                                continue
                            
                            print(Fore.YELLOW + "\nNow select the TEST pattern:")
                            test_pattern, test_values = select_pattern()
                            if test_pattern is None:
                                continue
                            
                            test_timesteps = get_test_timesteps()
                            model_name = get_model_name(trained_pattern, model_config["recon"], model_config["class"])
                            
                            results = test_single_pattern(
        			test_pattern, test_values,
        			model_name, model_config["recon"], model_config["class"], test_timesteps
    				)
                            
                            if results:
                                plot_trajectory(results, test_pattern, model_name, model_config["recon"], test_timesteps)
                            
                            input("\nPress ENTER to continue...")
                            
                        elif cl == "2":
                            # Test single model with all patterns
                            model_config = select_trained_timesteps()
                            if model_config is None:
                                continue
                            
                            trained_pattern, _ = select_pattern()
                            if trained_pattern is None:
                                continue
                            
                            test_timesteps = get_test_timesteps()

                            all_results = test_model_all_patterns(trained_pattern, model_config, test_timesteps)
                            
                            input("\nPress ENTER to continue...")
                            
                        elif cl == "3":
                                # Test all models with all patterns
	    			test_timesteps = get_test_timesteps()
    
    				total_models = len(AVAILABLE_MODELS) * len(PATTERNS)
    				confirm = input(Fore.YELLOW + f"\nThis will test {total_models} model-pattern combinations. Continue? (y/n): ").lower()
    				if confirm == 'y':
        				all_model_results = {}
        
        				with tqdm(total=total_models, desc="Testing All Models", unit="test") as pbar:
            					for model_key, model_config in AVAILABLE_MODELS.items():
                					for pattern_name in PATTERNS.keys():
                    						pbar.set_postfix_str(f"{model_key} - {pattern_name}")
                    
                    						model_name = get_model_name(pattern_name, model_config["recon"], model_config["class"])
                    
                    						model_results = {}
                    						for test_pattern_name, test_pattern_values in PATTERNS.items():
                        						results = test_single_pattern(
                            						test_pattern_name, test_pattern_values,
                            						model_name, model_config["recon"], model_config["class"], test_timesteps
                        						)
                        					if results:
                            						class_results = {}
                            						for cls_name in ["Square", "Random", "All-in", "All-out"]:
                                						if cls_name in results:
                                    							mean_probs = [np.mean(p) * 100 for p in results[cls_name]["predictions"]]
                                    							class_results[cls_name] = max(mean_probs)
                            						model_results[test_pattern_name] = class_results
                    
                    						all_model_results[f"{pattern_name}_{model_key}"] = model_results
                    
                    						# Generate bar plot for this model
                    						if model_results:
                        						plot_pattern_comparison_bar(
                            						model_results,
                            						model_name, model_config["recon"]
                        						)
                    
                    						pbar.update(1)
        
        				print_status("All evaluations complete!", "success")
    
    				input("\nPress ENTER to continue...")
                            
                        elif cl == "4":
                            # Grid search
                            model_config = select_trained_timesteps()
                            if trained_timesteps is None:
                                continue
                            
                            trained_pattern, _ = select_pattern()
                            if trained_pattern is None:
                                continue
                            
                            test_timesteps = get_test_timesteps()
                            
                            confirm = input(Fore.YELLOW + "\nGrid search may take a while. Continue? (y/n): ").lower()
                            if confirm == 'y':
                                run_grid_search(trained_pattern, model_config, test_timesteps)
                            
                            input("\nPress ENTER to continue...")
                            
                        elif cl == "0":
                            break
                        else:
                            print(Fore.RED + "Invalid option!")
                            input("Press ENTER...")
                            
                elif t == "0":
                    break
                else:
                    print(Fore.RED + "Invalid option!")
                    input("Press ENTER...")

        elif choice == "0":
            clear()
            banner("Goodbye!")
            break

        else:
            print(Fore.RED + "Invalid option!")
            input("Press ENTER...")

if __name__ == "__main__":
    run()
