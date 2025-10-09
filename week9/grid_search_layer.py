import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG_FILE = "configs/configtest.py"
LOG_DIR = "logs"
SUMMARY_DIR = "summaries"

# "mean" → take average accuracy across timesteps
# "max"  → take maximum accuracy across timesteps
ACCURACY_MODE = "mean"   # <---- change to "max" if desired

# Define gamma/beta patterns to inject
PATTERNS = {
    "Uniform": {"gamma": [0.33, 0.33, 0.33, 0.33], "beta": [0.33, 0.33, 0.33, 0.33]},
    "Gamma Increasing": {"gamma": [0.13, 0.33, 0.53, 0.33], "beta": [0.33, 0.33, 0.33, 0.33]},
    "Gamma Decreasing": {"gamma": [0.53, 0.33, 0.13, 0.33], "beta": [0.33, 0.33, 0.33, 0.33]},
    "Beta Increasing": {"gamma": [0.33, 0.33, 0.33, 0.33], "beta": [0.13, 0.33, 0.53, 0.33]},
    "Beta Decreasing": {"gamma": [0.33, 0.33, 0.33, 0.33], "beta": [0.53, 0.33, 0.13, 0.33]},
}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

print(f"\n{'=' * 50}")
print(f"Running CLASSIFICATION pattern experiments")
print(f"Accuracy Mode: {ACCURACY_MODE.upper()}")
print(f"Patterns to test: {list(PATTERNS.keys())}")
print(f"{'=' * 50}\n")


# ======================================================
# Helper functions
# ======================================================
def update_config(gamma_pattern, beta_pattern, pattern_name):
    """Update the config file with new gamma and beta patterns."""
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()
    with open(CONFIG_FILE, "w") as f:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("gammaset"):
                f.write(f"gammaset = [[{', '.join(f'{g:.2f}' for g in gamma_pattern)}]]  # {pattern_name}\n")
            elif stripped.startswith("betaset"):
                f.write(f"betaset = [[{', '.join(f'{b:.2f}' for b in beta_pattern)}]]  # {pattern_name}\n")
            else:
                f.write(line)


def run_experiment(pattern_name):
    """Run the experiment and extract mean or max accuracy across timesteps."""
    print(f"\n▶ Running pattern: {pattern_name}")

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    result = subprocess.run(
        ["python3", "main.py", "--config", "configtest"],
        capture_output=True,
        text=True,
        env=env
    )

    # Save log
    log_path = os.path.join(LOG_DIR, f"output_{pattern_name}.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)

    combined_output = result.stdout + "\n" + result.stderr

    # Extract timestep accuracies (global)
    timestep_accuracies = []
    for line in combined_output.splitlines():
        line = line.strip()
        match = re.match(r'^Timestep\s*\d+\s*[:\-]\s*([0-9]+\.[0-9]+)%', line)
        if match:
            timestep_accuracies.append(float(match.group(1)))

    if not timestep_accuracies:
        print("  ⚠️ No timestep accuracies found in output.")
        return {"Overall": 0.0}

    if ACCURACY_MODE == "mean":
        overall_acc = np.mean(timestep_accuracies)
    else:
        overall_acc = np.max(timestep_accuracies)

    print(f"  Overall Accuracy: {overall_acc:.2f}% ({ACCURACY_MODE})")
    return {"Overall": overall_acc}


# ======================================================
# EXPERIMENT LOOP
# ======================================================
results_per_pattern = {}

for pattern_name, vals in PATTERNS.items():
    gamma_pattern = vals["gamma"]
    beta_pattern = vals["beta"]
    update_config(gamma_pattern, beta_pattern, pattern_name)
    acc_dict = run_experiment(pattern_name)
    results_per_pattern[pattern_name] = acc_dict


# ======================================================
# BAR PLOT: Patterns on X-axis, Overall Accuracy as bar height
# ======================================================
patterns = list(PATTERNS.keys())
x = np.arange(len(patterns))
width = 0.5
values = [results_per_pattern[p]["Overall"] for p in patterns]

plt.figure(figsize=(10, 6))
plt.bar(x, values, width, color=plt.cm.tab10.colors[0])
plt.xticks(x, patterns, rotation=15)
plt.ylabel(f"{ACCURACY_MODE.capitalize()} Accuracy (%)")
plt.title(f"{ACCURACY_MODE.capitalize()} Accuracy per Pattern\nModel Trained with Classification for 1 Timestep")
plt.ylim(32,34)
plt.tight_layout()

save_path = os.path.join(SUMMARY_DIR, f"{ACCURACY_MODE}_accuracy_pattern_summary_classification.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\n✓ Saved summary plot as '{save_path}'")
print(f"✓ Logs saved in '{LOG_DIR}' directory.\n")

