import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ======================================================
# Configuration
# ======================================================
GAMMA_VALUES = np.arange(0.13, 0.54, 0.1)
BETA_VALUES = np.arange(0.13, 0.54, 0.1)
CONFIG_FILE = "configs/configilltest.py"
LOG_DIR = "logs"
HEATMAP_DIR = "heatmaps"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

print(f"\n{'*' * 50}")
print(f"Gamma Values Used: {GAMMA_VALUES}")
print(f"Beta Values Used: {BETA_VALUES}")
print(f"{'*' * 50}\n")

# ======================================================
# Helper functions
# ======================================================
def update_config(gamma, beta):
    """Update the config file with new gamma and beta values."""
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()

    with open(CONFIG_FILE, "w") as f:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("gammaset"):
                f.write(f"gammaset = [[{gamma:.2f}, {gamma:.2f}, {gamma:.2f}, {gamma:.2f}]]\n")
            elif stripped.startswith("betaset"):
                f.write(f"betaset = [[{beta:.2f}, {beta:.2f}, {beta:.2f}, {beta:.2f}]]\n")
            else:
                f.write(line)


def run_experiment(gamma, beta):
    """Run experiment and extract max accuracy per class across timesteps."""
    print(f"\nRunning: gamma={gamma:.2f}, beta={beta:.2f}")
    
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    result = subprocess.run(
        ["python3", "main.py", "--config", "configilltest"],
        capture_output=True,
        text=True,
        env=env
    )

    # Save output log
    log_path = os.path.join(LOG_DIR, f"output_gamma{gamma:.2f}_beta{beta:.2f}.log")
    with open(log_path, "w") as log_file:
        log_file.write(result.stdout)
        log_file.write("\n--- STDERR ---\n")
        log_file.write(result.stderr)

    # Parse output
    combined_output = result.stdout + "\n" + result.stderr
    class_accuracies = {}
    current_class = None

    for line in combined_output.splitlines():
        line = line.strip()
        print(line)

        # Detect class
        class_match = re.match(r'^Class:\s*([^,]+),', line)
        if class_match:
            current_class = class_match.group(1).strip()
            class_accuracies[current_class] = []
            continue

        # Detect timestep accuracy
        if current_class:
            timestep_match = re.match(r'^Timestep\s*\d+\s*:\s*([0-9]+\.[0-9]+)%', line)
            if timestep_match:
                acc = float(timestep_match.group(1))
                class_accuracies[current_class].append(acc)

    # Compute max accuracy per class
    max_acc_per_class = {cls: max(accs) if accs else 0.0 for cls, accs in class_accuracies.items()}

    #max_acc_per_class = {cls: sum(accs)/len(accs) if accs else 0.0 for cls, accs in class_accuracies.items()}
    for cls, acc in max_acc_per_class.items():
        print(f"Max accuracy for class {cls}: {acc:.2f}%")

    return max_acc_per_class


# ======================================================
# Grid search
# ======================================================
all_classes = set()
results_per_class = {}  # {class_name: {(gamma,beta): max_acc}}

total = len(GAMMA_VALUES) * len(BETA_VALUES)
count = 0

for gamma in GAMMA_VALUES:
    for beta in BETA_VALUES:
        count += 1
        print(f"\nExperiment {count}/{total}")
        update_config(gamma, beta)
        max_acc_dict = run_experiment(gamma, beta)

        # Track all classes
        for cls, acc in max_acc_dict.items():
            all_classes.add(cls)
            results_per_class.setdefault(cls, {})[(gamma, beta)] = acc


# ======================================================
# Construct heatmaps per class
# ======================================================
for cls in all_classes:
    matrix = np.zeros((len(BETA_VALUES), len(GAMMA_VALUES)))
    for i, beta in enumerate(BETA_VALUES):
        for j, gamma in enumerate(GAMMA_VALUES):
            matrix[i, j] = results_per_class[cls].get((gamma, beta), 0.0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=[f"{g:.2f}" for g in GAMMA_VALUES],
        yticklabels=[f"{b:.2f}" for b in BETA_VALUES],
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        cbar_kws={"label": f"Max Accuracy (%) - Class {cls}"}
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Gamma", fontsize=14)
    plt.ylabel("Beta", fontsize=14)
    plt.title(f"Max Accuracy Heatmap for Class '{cls}' With Illusion Training on 1 Timesteps", fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(HEATMAP_DIR, f"heatmap_{cls}.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"✓ Heatmap for class '{cls}' saved as '{heatmap_path}'")


# ======================================================
# Print best configurations per class
# ======================================================
for cls in all_classes:
    matrix = np.zeros((len(BETA_VALUES), len(GAMMA_VALUES)))
    for i, beta in enumerate(BETA_VALUES):
        for j, gamma in enumerate(GAMMA_VALUES):
            matrix[i, j] = results_per_class[cls].get((gamma, beta), 0.0)
    best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_gamma = GAMMA_VALUES[best_idx[1]]
    best_beta = BETA_VALUES[best_idx[0]]
    best_acc = matrix[best_idx]

    print(f"\nBest configuration for class '{cls}':")
    print(f"Gamma = {best_gamma:.2f}")
    print(f"Beta  = {best_beta:.2f}")
    print(f"Max Accuracy = {best_acc:.2f}%")

print("\nAll logs saved in 'logs/' directory.")
print("All heatmaps saved in 'heatmaps/' directory.")

