# main.py

import importlib
import sys
import argparse
import os

def load_config(config_name):
    return importlib.import_module(config_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Add config directory to Python path
    sys.path.append(os.path.abspath("configs"))

    config = load_config(args.config)

    # Now use config variables
    print(f"Running with config: {args.config}")
    print(f"Batch size: {config.batch_size}, LR: {config.lr}, Epochs: {config.epochs}")
    # your training logic here

