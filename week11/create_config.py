import os
import sys
import itertools
from model_tracking import get_tracker
from utils import generate_model_name


BASE_DIR = "configs/base_config.py"
CONFIG_FILE = BASE_DIR


def update_config(
    gamma_pattern,
    beta_pattern,
    pattern_name,
    model_name,
    timesteps,
    train_cond,
    last_neurons,
    seed,
    lr,
    epochs,
    base_recon_model=None,
    checkpoint_epoch=None,
    classification_dataset=None,
    reconstruction_dataset=None
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

            elif stripped.startswith("epochs"):
                f.write(f'epochs = {epochs[0]}\n')

            elif stripped.startswith("classification_datasetpath"):
                f.write(f'classification_datasetpath = "{classification_dataset}"\n')

            elif stripped.startswith("recon_datasetpath"):
                f.write(f'recon_datasetpath = "{reconstruction_dataset}"\n')

            elif stripped.startswith("seed"):
                f.write(f"seed = {seed}\n")

            elif stripped.startswith("lr"):
                f.write(f"lr = {lr}\n")

            elif stripped.startswith("timesteps"):
                f.write(f"timesteps = {timesteps[0] if isinstance(timesteps, list) else timesteps}\n")

            elif stripped.startswith("classification_neurons"):
                f.write(f"classification_neurons = {last_neurons}\n")

            elif stripped.startswith("experiment_name"):
                f.write(
                    f'experiment_name = "Testing {model_name} with {pattern_name} '
                    f'pattern at {timesteps} timesteps"\n'
                )

            elif stripped.startswith("training_condition"):
                if train_cond is None:
                    f.write(f"training_condition = {train_cond}\n")
                else:
                    f.write(f'training_condition = "{train_cond}"\n')
            
            # Add classification-specific fields
            elif train_cond == "classification_training_shapes":
                if "# Classification training fields" in line:
                    f.write(f"base_recon_model = \"{base_recon_model}\"\n")
                    f.write(f"checkpoint_epoch = {checkpoint_epoch}\n")
                else:
                    f.write(line)
            else:
                f.write(line)


def create_config_files(
    seeds,
    patterns,
    train_cond,
    epochs,
    lr_list,
    timesteps,
    number_of_classes,
    base_recon_models=None,
    checkpoint_epochs=None,
    dataset_list=None
):
    """
    Create multiple config files from parameters for batch experiments.
    Returns list of config file paths and their associated model names.
    """

    config_paths = []
    model_names = []

    global CONFIG_FILE

    exp_id = 0

    # Pattern definitions
    PATTERNS = {
        "Uniform": {
            "gamma": [0.33, 0.33, 0.33, 0.33],
            "beta": [0.33, 0.33, 0.33, 0.33],
        },
        "Gamma Increasing": {
            "gamma": [0.13, 0.33, 0.53, 0.33],
            "beta": [0.33, 0.33, 0.33, 0.33],
        },
        "Gamma Decreasing": {
            "gamma": [0.53, 0.33, 0.13, 0.33],
            "beta": [0.33, 0.33, 0.33, 0.33],
        },
        "Beta Increasing": {
            "gamma": [0.33, 0.33, 0.33, 0.33],
            "beta": [0.13, 0.33, 0.53, 0.33],
        },
        "Beta Decreasing": {
            "gamma": [0.33, 0.33, 0.33, 0.33],
            "beta": [0.53, 0.33, 0.13, 0.33],
        },
        "Beta Inc & Gamma Dec": {
            "gamma": [0.53, 0.33, 0.13, 0.33],
            "beta": [0.13, 0.33, 0.53, 0.33],
        },
    }

    # Create configs directory
    os.makedirs("configs", exist_ok=True)
    
    tracker = get_tracker()

    # If classification training, iterate over base models too
    if train_cond == "classification_training_shapes" and base_recon_models:
        for base_model in base_recon_models:
            # FIX 1: Extract dataset from base recon model
            base_model_info = tracker.get_model(base_model)
            if not base_model_info:
                print(f"Warning: Base model {base_model} not found in registry!")
                continue
            
            # Get the dataset the base model was trained on
            base_dataset = base_model_info['config'].get('Dataset', 'custom_illusion_dataset')
            
            # FIX 2: Extract seed from base model
            base_seed = base_model_info['config'].get('seed', 42)
            
            # Determine number of classes based on dataset
            if base_dataset == "custom_illusion_dataset":
                num_classes = 6
            elif base_dataset in ["cifar10", "stl10"]:
                num_classes = 10
            else:
                num_classes = 10  # default
            
            for checkpoint_epoch in checkpoint_epochs:
                # Use the base model's seed instead of iterating through seeds
                for pattern, lr, timestep in itertools.product(
                    patterns, lr_list, timesteps
                ):
                    base_model_with_chk = f"{base_model}_chk{checkpoint_epoch}"
                   
                    model_name = generate_model_name(
                         pattern=pattern,
                         seed=base_seed,  # Use base model's seed
                         train_cond=train_cond,
                         recon_timesteps=timestep,
                         classification_timesteps=timestep,
                         dataset=base_dataset,
                         base_model=base_model_with_chk
                    )

                    config_dict = {
                        "pattern": pattern,
                        "seed": base_seed,  # Use base model's seed
                        "train_cond": train_cond,
                        "lr": lr,
                        "timesteps": timestep,
                        "last_neurons": num_classes,
                        "epochs": epochs[0] if isinstance(epochs, list) else epochs,
                        "base_recon_model": base_model,
                        "checkpoint_epoch": checkpoint_epoch,
                        "dataset": base_dataset
                    }
                    tracker.register_model(model_name, config_dict)

                    gamma_pattern = PATTERNS[pattern]["gamma"]
                    beta_pattern = PATTERNS[pattern]["beta"]

                    cfg_path = f"configs/config_{exp_id}.py"
                    cfg_command = f"config_{exp_id}"
                    
                    CONFIG_FILE = cfg_path

                    with open(BASE_DIR) as f:
                        base = f.read()

                    # Add classification-specific fields if needed
                    if train_cond == "classification_training_shapes":
                        base += "\n# Classification training fields\n"

                    with open(cfg_path, "w") as f:
                        f.write(base)

                    update_config(
                        gamma_pattern,
                        beta_pattern,
                        pattern,
                        model_name,
                        timestep,
                        train_cond,
                        num_classes,
                        base_seed,  # Use base model's seed
                        lr,
                        epochs,
                        base_recon_model=base_model,
                        checkpoint_epoch=checkpoint_epoch,
                        classification_dataset=base_dataset,
                        reconstruction_dataset=base_dataset
                    )

                    config_paths.append(cfg_command)
                    model_names.append(model_name)
                    exp_id += 1
    else:
        # Reconstruction training (original logic)
        for seed, pattern, lr, timestep, dataset in itertools.product(
            seeds, patterns, lr_list, timesteps, dataset_list
        ):
            model_name = generate_model_name(
                pattern=pattern,
                seed=seed,
                train_cond=train_cond,
                recon_timesteps=timestep,
                dataset=dataset
            )

            config_dict = {
                "pattern": pattern,
                "seed": seed,
                "train_cond": train_cond,
                "lr": lr,
                "timesteps": timestep,
                "last_neurons": number_of_classes[0],
                "Dataset": dataset,
                "epochs": epochs[0] if isinstance(epochs, list) else epochs,
            }
            tracker.register_model(model_name, config_dict)

            gamma_pattern = PATTERNS[pattern]["gamma"]
            beta_pattern = PATTERNS[pattern]["beta"]

            cfg_path = f"configs/config_{exp_id}.py"
            cfg_command = f"config_{exp_id}"
            
            CONFIG_FILE = cfg_path

            with open(BASE_DIR) as f:
                base = f.read()

            with open(cfg_path, "w") as f:
                f.write(base)

            update_config(
                gamma_pattern,
                beta_pattern,
                pattern,
                model_name,
                timestep,
                train_cond,
                number_of_classes[0],
                seed,
                lr,
                epochs,
                classification_dataset=dataset,
                reconstruction_dataset=dataset
            )

            config_paths.append(cfg_command)
            model_names.append(model_name)
            exp_id += 1

    print(f"Generated {len(config_paths)} config files")
    print(f"Registered {len(model_names)} models in tracker")

    return config_paths, model_names
