import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ======================================================
# Configuration
# ======================================================
GAMMA_VALUES = np.arange(0.13, 0.53, 0.1)
BETA_VALUES = np.arange(0.13, 0.53, 0.1)
file_version = "configilltest10"
CONFIG_FILE = f"configs/{file_version}.py"
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
                f.write(f"gammaset = [[0.33,0.33,{gamma:.2f},0.33]]\n")
            elif stripped.startswith("betaset"):
                f.write(f"betaset = [[0.33,0.33,{beta:.2f},0.33]]\n")
            else:
                f.write(line)


def run_experiment(gamma, beta):
    """Run experiment and extract max accuracy per class."""
    print(f"\nRunning: gamma={gamma:.2f}, beta={beta:.2f}")

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    result = subprocess.run(
        ["python3", "main.py", "--config", f"{file_version}"],
        capture_output=True,
        text=True,
        env=env
    )

    # Save output
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

        # Detect class
        class_match = re.match(r"^Class:\s*([^,]+),", line)
        if class_match:
            current_class = class_match.group(1).strip()
            class_accuracies[current_class] = []
            continue

        # Detect timestep accuracy
        if current_class:
            timestep_match = re.match(r"^Timestep\s*\d+:\s*([0-9]+\.[0-9]+)%", line)
            if timestep_match:
                acc = float(timestep_match.group(1))
                class_accuracies[current_class].append(acc)

    # Compute max accuracy per class
    max_acc_per_class = {cls: max(accs) if accs else 0.0 for cls, accs in class_accuracies.items()}

    for cls, acc in max_acc_per_class.items():
        print(f"  Max accuracy for class {cls}: {acc:.2f}%")

    return max_acc_per_class


# ======================================================
# Grid search and store results
# ======================================================
results = {}  # {(gamma,beta): {class: max_acc}}
total = len(GAMMA_VALUES) * len(BETA_VALUES)
count = 0

for gamma in GAMMA_VALUES:
    for beta in BETA_VALUES:
        count += 1
        print(f"\n{'='*50}")
        print(f"Experiment {count}/{total}")
        print(f"{'='*50}")
        update_config(gamma, beta)
        max_acc = run_experiment(gamma, beta)
        results[(gamma, beta)] = max_acc


# ======================================================
# Compute Illusion Index per (gamma,beta)
# ======================================================
print(f"\n{'='*50}")
print("Computing Illusion Index")
print(f"{'='*50}\n")

illusion_matrix = np.zeros((len(BETA_VALUES), len(GAMMA_VALUES)))

for i, beta in enumerate(BETA_VALUES):
    for j, gamma in enumerate(GAMMA_VALUES):
        res = results.get((gamma, beta), {})
        
        # Get max accuracies (already as percentages, e.g., 93.85)
        max_allin = res.get("All-in", 0)
        max_allout = res.get("All-out", 0)
        max_random = res.get("Random", 0)
        max_square = res.get("Square", 0)
        
        # Calculate denominator: (all-out + random)/2
        denom = (max_allout + max_random) / 2
        
        # Illusion Index = all-in / ((all-out + random)/2)
        if denom > 0:
            illusion_index = max_allin / denom
        else:
            illusion_index = 0
        
        print(f"γ={gamma:.2f}, β={beta:.2f}:")
        print(f"  Square={max_square:.2f}%, All-in={max_allin:.2f}%, "
              f"All-out={max_allout:.2f}%, Random={max_random:.2f}%")
        print(f"  Denominator={denom:.2f}, Illusion Index={illusion_index:.3f}\n")
        
        illusion_matrix[i, j] = illusion_index


# ======================================================
# Plot single heatmap for Illusion Index
# ======================================================
print(f"{'='*50}")
print("Generating Heatmap")
print(f"{'='*50}\n")

plt.figure(figsize=(10, 8))
sns.heatmap(
    illusion_matrix,
    xticklabels=[f"{g:.2f}" for g in GAMMA_VALUES],
    yticklabels=[f"{b:.2f}" for b in BETA_VALUES],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0,
    vmax=1.6,  # Auto-scale based on data
    cbar_kws={"label": "Illusion Index"}
)
plt.gca().invert_yaxis()
plt.xlabel("Gamma", fontsize=14)
plt.ylabel("Beta", fontsize=14)
plt.title("Illusion Index for Modification of Third Layer", fontsize=13)
plt.tight_layout()

heatmap_path = os.path.join(HEATMAP_DIR, "illusion_index_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"✓ Illusion index heatmap saved to '{heatmap_path}'")

# ======================================================
# Save results summary
# ======================================================
summary_path = os.path.join(LOG_DIR, "results_summary.txt")
with open(summary_path, "w") as f:
    f.write("Grid Search Results Summary\n")
    f.write("="*70 + "\n\n")
    
    for i, beta in enumerate(BETA_VALUES):
        for j, gamma in enumerate(GAMMA_VALUES):
            res = results.get((gamma, beta), {})
            f.write(f"γ={gamma:.2f}, β={beta:.2f}:\n")
            f.write(f"  Square:  {res.get('Square', 0):.2f}%\n")
            f.write(f"  All-in:  {res.get('All-in', 0):.2f}%\n")
            f.write(f"  All-out: {res.get('All-out', 0):.2f}%\n")
            f.write(f"  Random:  {res.get('Random', 0):.2f}%\n")
            f.write(f"  Illusion Index: {illusion_matrix[i, j]:.3f}\n")
            f.write("-"*70 + "\n")

print(f"✓ Results summary saved to '{summary_path}'")
print("\nAll done!")
