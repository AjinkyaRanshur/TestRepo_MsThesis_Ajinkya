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
CONFIG_FILE = "configs/configtest.py"
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

print(f"\n{'*' * 50}")
print(f"Gamma Values Used: {GAMMA_VALUES} (Type: {type(GAMMA_VALUES)})")
print(f"Beta Values Used: {BETA_VALUES} (Type: {type(BETA_VALUES)})")
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
    """Run the experiment for given gamma/beta and extract accuracy."""
    print(f"\nRunning: gamma={gamma:.2f}, beta={beta:.2f}")
    
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    result = subprocess.run(
        ["python3", "main.py", "--config", "configtest"],
        capture_output=True,
        text=True,
        env=env
    )

    # Save output log for debugging
    log_path = os.path.join(LOG_DIR, f"output_gamma{gamma:.2f}_beta{beta:.2f}.log")
    with open(log_path, "w") as log_file:
        log_file.write(result.stdout)
        log_file.write("\n--- STDERR ---\n")
        log_file.write(result.stderr)

    # Extract accuracies from stdout
    # Combine stdout and stderr for parsing
    combined_output = result.stdout + "\n" + result.stderr

    accuracies = []
    for line in combined_output.splitlines():
        print(line)
        if "Timestep" in line and "%" in line:
            match = re.search(r'Timestep\s*\d+\s*[:\-]\s*([0-9]+\.[0-9]+)%', line)
            if match:
                accuracies.append(float(match.group(1)))


    #max_acc = max(accuracies) if accuracies else 0.0
    mean_acc= sum(accuracies)/len(accuracies) if accuracies else 0.0
    #print(f"Max accuracy: {max_acc:.2f}%")
    print(f"Mean accuracy: {mean_acc:.2f}%")
    return mean_acc


# ======================================================
# Grid search
# ======================================================
results = {}
total = len(GAMMA_VALUES) * len(BETA_VALUES)
count = 0

for gamma in GAMMA_VALUES:
    for beta in BETA_VALUES:
        count += 1
        print(f"\nExperiment {count}/{total}")
        update_config(gamma, beta)
        acc = run_experiment(gamma, beta)
        results[(gamma, beta)] = acc

# ======================================================
# Construct results matrix
# ======================================================
matrix = np.zeros((len(BETA_VALUES), len(GAMMA_VALUES)))
for i, beta in enumerate(BETA_VALUES):
    for j, gamma in enumerate(GAMMA_VALUES):
        matrix[i, j] = results.get((gamma, beta), 0.0)

# ======================================================
# Plot heatmap
# ======================================================
plt.figure(figsize=(12, 10))
sns.heatmap(
    matrix,
    xticklabels=[f"{g:.2f}" for g in GAMMA_VALUES],
    yticklabels=[f"{b:.2f}" for b in BETA_VALUES],
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    cbar_kws={"label": "Mean Accuracy (%)"}
)
plt.gca().invert_yaxis()  # 👈 reverses y-axis order
plt.xlabel("Gamma", fontsize=14)
plt.ylabel("Beta", fontsize=14)
plt.title("Mean Accuracy Across Timesteps With Model Trained on 1 timesteps Using Classification", fontsize=16)
plt.tight_layout()
plt.savefig("accuracy_heatmap.png", dpi=300)
print("\n✓ Heatmap saved as 'accuracy_heatmap.png'")

# ======================================================
# Print best configuration
# ======================================================
best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
best_gamma = GAMMA_VALUES[best_idx[1]]
best_beta = BETA_VALUES[best_idx[0]]
best_acc = matrix[best_idx]

print(f"\nBest configuration:")
print(f"Gamma = {best_gamma:.2f}")
print(f"Beta  = {best_beta:.2f}")
print(f"Mean Accuracy = {best_acc:.2f}%")
print("\nAll logs saved in 'logs/' directory.")


