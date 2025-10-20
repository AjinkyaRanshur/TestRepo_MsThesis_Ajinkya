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

# ======================================================
# COMBINED INTEGRATED PLOT: Accuracy Bars + Gamma/Beta Trajectories
# ======================================================
from matplotlib.patches import FancyArrowPatch

patterns = list(PATTERNS.keys())
x = np.arange(len(patterns))
width = 0.5
values = [results_per_pattern[p]["Overall"] for p in patterns]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]})

# --------------------------
# (1) Accuracy Bar Chart
# --------------------------
ax1.bar(x, values, width, color=plt.cm.tab10.colors[0])
ax1.set_xticks([])
ax1.set_ylabel(f"{ACCURACY_MODE.capitalize()} Accuracy (%)")
ax1.set_ylim(min(values) - 0.5, max(values) + 0.5)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Optional: show numeric values above bars
for i, v in enumerate(values):
    ax1.text(x[i], v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

# --------------------------
# (2) Gamma/Beta Pattern Arrows (Aligned Below Bars)
# --------------------------
offset = 0.12
arrow_color_gamma = 'tab:blue'
arrow_color_beta = 'tab:red'

# Reference lines for clarity
ax2.hlines([0.13, 0.33, 0.53], -0.5, len(patterns)-0.5, linestyles='dotted', alpha=0.25)

for i, pattern in enumerate(patterns):
    vals = PATTERNS[pattern]
    gammas = vals["gamma"]
    betas = vals["beta"]

    # For each consecutive pair, draw downward arrow (ignore final value)
    gx_center = x[i] - offset
    bx_center = x[i] + offset

    for j in range(len(gammas) - 1):
        y1, y2 = gammas[j], gammas[j + 1]
        arrow_gamma = FancyArrowPatch((gx_center, y1), (gx_center, y2),
                                      arrowstyle='-|>', mutation_scale=12,
                                      linewidth=2.0, color=arrow_color_gamma)
        ax2.add_patch(arrow_gamma)

    for j in range(len(betas) - 1):
        y1, y2 = betas[j], betas[j + 1]
        arrow_beta = FancyArrowPatch((bx_center, y1), (bx_center, y2),
                                     arrowstyle='-|>', mutation_scale=12,
                                     linewidth=2.0, color=arrow_color_beta)
        ax2.add_patch(arrow_beta)

# Cosmetics for ax2
ax2.set_xticks(x)
ax2.set_xticklabels(patterns, rotation=20, ha='right')
ax2.set_xlim(-0.6, len(patterns)-0.4)
ax2.set_ylim(0.08, 0.58)
ax2.set_ylabel("γ / β Value")
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.set_yticks([0.13, 0.33, 0.53])
ax2.legend(handles=[
    FancyArrowPatch((0,0), (0,0), color=arrow_color_gamma, label='γ'),
    FancyArrowPatch((0,0), (0,0), color=arrow_color_beta, label='β')
], loc='upper right')

# No titles
ax1.set_title("")
ax2.set_title("")

plt.tight_layout(h_pad=0.5)
save_path_combined = os.path.join(SUMMARY_DIR, f"{ACCURACY_MODE}_accuracy_and_trajectories_combined.png")
plt.savefig(save_path_combined, dpi=300)
plt.close()

print(f"\n✓ Saved integrated bar+pattern plot as '{save_path_combined}'")
print(f"✓ Logs saved in '{LOG_DIR}' directory.\n")


