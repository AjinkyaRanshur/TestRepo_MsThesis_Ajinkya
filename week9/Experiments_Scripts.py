import sys
import os
from main import load_config, main
import pyfiglet

def bar_plots():

    return None

def traj_plots():

    return None

def traj_plots_ill():

    return None

def run_and_analyze(config_name):
    # Add config directory to Python path
    sys.path.append(os.path.abspath("configs"))
    config=load_config(config_name)
    results=main(config)

    return results

if __name__ == "__main__":
    f = pyfiglet.figlet_format("Pred-Net Analyzer",font="ogre")
    print("="*50)
    print(f)
    print("="*50)
    print("1.) Analyze the Trajectories over Timesteps")
    print("2.) Perform Pattern Testing on The Model")
    selection = int(input("Please choose one of the menu options.\n>> "))

    results=run_and_analyze("configilltest")

    if selection ==1:
        print(results)
















    

