import os
import pyfiglet
from colorama import Fore, Style, init
import sys
from main import load_config, main
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

# init colorama for Windows
init(autoreset=True)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG_FILE = "configs/configfile.py"
CONFIG_MODULE = "configfile"


# Pattern definitions
PATTERNS = {
    "Uniform": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Gamma Increasing": {
        "gamma": [0.13, 0.33, 0.53, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Gamma Decreasing": {
        "gamma": [0.53, 0.33, 0.13, 0.33],
        "beta": [0.33, 0.33, 0.33, 0.33]
    },
    "Beta Increasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33]
    },
    "Beta Decreasing": {
        "gamma": [0.33, 0.33, 0.33, 0.33],
        "beta": [0.53, 0.33, 0.13, 0.33]
    },
    "Beta Inc & Gamma Dec": {
        "gamma": [0.53, 0.33, 0.13, 0.33],
        "beta": [0.13, 0.33, 0.53, 0.33]
    }
}


























