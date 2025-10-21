import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG_FILE = "configs/configilltest.py"
LOG_DIR = "logs"
SUMMARY_DIR = "summaries"

# "mean" → take average accuracy across timesteps
# "max"  → take max accuracy across timesteps
ACCURACY_MODE = "mean"   # <---- change this to "mean" if desired

# Predefined gamma/beta patterns to test
PATTERNS = {
    "Uniform": {"gamma": [0.33, 0.33, 0.33, 0.33], "beta": [0.33, 0.33, 0.33, 0.33]},
    "Gamma Increasing": {"gamma": [0.13, 0.33, 0.53, 0.33], "beta": [0.33, 0.33, 0.33, 0.33]},
    "Gamma Decreasing": {"gamma": [0.53, 0.33, 0.13, 0.33], "beta": [0.33, 0.33, 0.33, 0.33]},
    "Beta Increasing": {"gamma": [0.33, 0.33, 0.33, 0.33], "beta": [0.13, 0.33, 0.53, 0.33]},
    "Beta Decreasing": {"gamma": [0.33, 0.33, 0.33, 0.33], "beta": [0.53, 0.33, 0.13, 0.33]},
    "Beta Inc. & Gamma Dec.": {"gamma": [0.53, 0.33, 0.13, 0.33], "beta": [0.13, 0.33, 0.53, 0.33]}
}

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

print(f"\n{'=' * 50}")
print(f"Running experiment with ACCURACY_MODE = '{ACCURACY_MODE.upper()}'")
print(f"Testing {len(PATTERNS)} gamma/beta patterns:\n")
for k, v in PATTERNS.items():
    print(f"  • {k}: gamma={v['gamma']}, beta={v['beta']}")
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
                f.write(f"gammaset = [[{', '.join(f'{g:.2f}' for g in gamma_pattern)}]]  # pattern: {pattern_name}\n")
            elif stripped.startswith("betaset"):
                f.write(f"betaset = [[{', '.join(f'{b:.2f}' for b in beta_pattern)}]]  # pattern: {pattern_name}\n")
            else:
                f.write(line)


def run_experiment(pattern_name):
    """Run the experiment and extract mean or max accuracy per class."""
    print(f"\n▶ Running pattern: {pattern_name}")

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    result = subprocess.run(
        ["python3", "main.py", "--config", "configilltest"],
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

    # Combine stdout + stderr for parsing
    combined_output = result.stdout + "\n" + result.stderr

    # ======================================================
    # PARSING SECTION
    # ======================================================
    class_accuracies = {}
    current_class = None

    for line in combined_output.splitlines():
        line = line.strip()

        # --- Detect new class ---
        if line.startswith("Class:"):
            match = re.search(r"Class:\s*([^,]+)", line)
            if match:
                current_class = match.group(1).strip()
                class_accuracies[current_class] = []
            continue

        # --- Detect timestep accuracy ---
        if current_class is not None:
            match = re.search(r"Timestep\s*\d+\s*:\s*([0-9]+(?:\.[0-9]+)?)%", line)
            if match:
                acc = float(match.group(1))
                class_accuracies[current_class].append(acc)

    # ======================================================
    # COMPUTE MEAN OR MAX
    # ======================================================
    if ACCURACY_MODE == "mean":
        acc_dict = {
            cls: np.mean(accs) if accs else 0.0
            for cls, accs in class_accuracies.items()
        }
    else:
        acc_dict = {
            cls: max(accs) if accs else 0.0
            for cls, accs in class_accuracies.items()
        }

    # ======================================================
    # PRINT SUMMARY
    # ======================================================
    print("\n📊 Summary of parsed results:")
    for cls, acc in acc_dict.items():
        n = len(class_accuracies[cls])
        print(f"  {cls:<10}: {acc:.2f}% ({ACCURACY_MODE}, {n} timesteps)")

    return acc_dict



# ======================================================
# EXPERIMENT LOOP
# ======================================================
all_classes = set()
results_per_pattern = {}

for pattern_name, vals in PATTERNS.items():
    gamma_pattern = vals["gamma"]
    beta_pattern = vals["beta"]
    update_config(gamma_pattern, beta_pattern, pattern_name)
    acc_dict = run_experiment(pattern_name)

    for cls, acc in acc_dict.items():
        all_classes.add(cls)
    results_per_pattern[pattern_name] = acc_dict


# ======================================================
# BAR PLOT (patterns on X-axis, classes as colors)
# ======================================================
patterns = list(PATTERNS.keys())
classes = sorted(list(all_classes))

x = np.arange(len(patterns))
width = 0.12
colors = plt.cm.tab10.colors

plt.figure(figsize=(12, 6))
for i, cls in enumerate(classes):
    cls_values = [results_per_pattern[p].get(cls, 0.0) for p in patterns]
    plt.bar(x + i * width, cls_values, width, label=cls, color=colors[i % len(colors)])

plt.xticks(x + width * (len(classes) - 1) / 2, patterns, rotation=15)
plt.ylabel(f"{ACCURACY_MODE.capitalize()} Accuracy (%)")
plt.title(f"{ACCURACY_MODE.capitalize()} Accuracy per Pattern on Models Trained with 10 Timestep for the Illusion Dataset")
plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.ylim(0,100)
plt.tight_layout()
plt.savefig(os.path.join(SUMMARY_DIR, f"{ACCURACY_MODE}_accuracy_pattern_summary.png"), dpi=300)
plt.close()

print(f"\n✓ Saved bar plot in '{SUMMARY_DIR}' as '{ACCURACY_MODE}_accuracy_pattern_summary.png'")
print(f"✓ All logs saved in '{LOG_DIR}' directory.\n")

