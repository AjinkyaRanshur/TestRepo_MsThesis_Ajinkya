import os
import sys


def update_config(
    gamma_pattern,
    beta_pattern,
    pattern_name,
    model_name,
    timesteps,
    iterations,
    train_cond,
    datasetpath
):
    """Update the config file with new parameters."""
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()

    with open(CONFIG_FILE, "w") as f:
        for line in lines:
            stripped = line.strip()

            if stripped.startswith("gammaset"):
                f.write(
                    f"gammaset = [[{', '.join(f'{g:.2f}' for g in gamma_pattern)}]]  "
                    f"# pattern: {pattern_name}\n"
                )

            elif stripped.startswith("betaset"):
                f.write(
                    f"betaset = [[{', '.join(f'{b:.2f}' for b in beta_pattern)}]]  "
                    f"# pattern: {pattern_name}\n"
                )

            elif stripped.startswith("model_name"):
                f.write(f'model_name = "{model_name}"\n')

            elif stripped.startswith("timesteps"):
                f.write(f"timesteps = {timesteps}\n")

            elif stripped.startswith("iterations"):
                f.write(f"iterations = {iterations}\n")

            elif stripped.startswith("experiment_name"):
                f.write(
                    f'experiment_name = "Testing {model_name} with {pattern_name} '
                    f'pattern at {timesteps} timesteps"\n'
                )

            elif stripped.startswith("training_condition"):
              if train_cond == None:
                f.write(f'training_condition = {train_cond}\n')
              else:
                f.write(f'training_condition = "{train_cond}"\n')

            elif stripped.startswith("datasetpath"):
                f.write(f'datasetpath = "{datasetpath}"\n')

            else:
                f.write(line)

#USED TO CREATE MULTIPLE CONFIG FILES FROM JSON FILE IN CASE WE NEED TO PERFORM MULTIPLE EXPERIMENTS
def create_config_files()


return None











