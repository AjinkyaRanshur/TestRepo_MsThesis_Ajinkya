from utils import get_model_name
from 


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
        f"Iterations: {iterations}"
	f"Recon Timesteps:{recon_timesteps}",
	f"Class Timesteps:{class_timesteps}"
    ])
    
    try:
        print_status("Updating configuration...", "running")
        update_config(gamma_pattern, beta_pattern, pattern_name, model_name, class_timesteps, iterations,"illusion_train","data/visual_illusion_dataset")
        
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

    with tqdm(
        total=len(PATTERNS),
        desc="Training Patterns",
        unit="pattern",
        bar_format="{l_bar}{bar:30}{r_bar}"
    ) as pbar:

        for idx, (pattern_name, pattern_values) in enumerate(PATTERNS.items(), 1):
            pbar.set_postfix_str(f"Current: {pattern_name}")
            result = train_single_pattern(
                pattern_name,
                pattern_values,
                timesteps,
                timesteps,iterations
            )
            results[pattern_name] = result
            pbar.update(1)

    # Summary
    print("\n")
    summary_lines = []
    for name, result in results.items():
        status = " Success" if result is not None else "X Failed"
        summary_lines.append(f"{name}: {status}")

    print_box("TRAINING SUMMARY", summary_lines, Fore.GREEN)
    return results
