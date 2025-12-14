import os
import pyfiglet
from colorama import Fore, Style, init


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def banner(text):
    print(Fore.CYAN + pyfiglet.figlet_format(text, font="ogre"))

def parse_list(x, cast=int):
    return [cast(v.strip()) for v in x.split(",")]

def generate_model_name(pattern,seed,train_cond,recon_timesteps,classification_timesteps=10):
	if train_cond == "recon_pc_train":
		model_name=f"pc_recon{recon_timesteps}_{pattern}_seed{seed}"
		return model_name
	elif train_cond == "classification_training_shapes":
		model_name=f"pc_recon{recon_timesteps}_class{classification_timesteps}_{pattern}_seed{seed}"
		return model_name


