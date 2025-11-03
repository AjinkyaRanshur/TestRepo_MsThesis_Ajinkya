import sys
import os
from main import load_config, main
import pyfiglet
import numpy as np
import matplotlib.pyplot as plt

def bar_plots():

    return None

def traj_plots(xaxis,yaxis,title_label,accuracy_data,prefix):
    plt.figure(figsize=(12,6))
    plt.plot(range(len(accuracy_data)), accuracy_data, linewidth=2, markersize=6)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.ylim(0,100)
    plt.title(title_label)
    plt.grid(False)
    plt.savefig(rf'result_folder/Timepass.png',dpi=300)

    return True

def traj_plots_ill(xaxis,yaxis,title_label,accuracy_data,prefix):
    plt.figure(figsize=(12,6))
    for class_name,data in accuracy_data.items():
        plt.plot(data['timesteps'],data['values'],linestyle='-',label=class_name)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.ylim(0,100)
    plt.title(title_label)
    plt.grid(False)
    plt.legend()
    #plt.savefig(rf'result_folder/Trajetories_Of_Model_Trained_with_just_reconstruction.png',dpi=300)
    plt.savefig(rf'result_folder/Trajetories_Of_Model_Trained_{prefix}_pc_dynamics.png',dpi=300)

    return True

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

    results=run_and_analyze("configilltest10")
    add_prefix_title="with"

    if selection ==1:
        accuracy_data={}
        for cls_name in ["Square", "Random", "All-in", "All-out"]:
            accuracy_data[cls_name]={'timesteps':[],'values':[]} 

        for cls_name in ["Square", "Random", "All-in", "All-out"]:
            total = results[cls_name]["total"]
            mean_probs = [np.mean(p) * 100 for p in results[cls_name]["predictions"]]
            for t, acc in enumerate(mean_probs):
                accuracy_data[cls_name]['timesteps'].append(int(t))
                accuracy_data[cls_name]['values'].append(float(acc))

        train_bool=traj_plots_ill("Timesteps","Probability Of a Being a Square",f"Performance of Model Trained {add_prefix_title} PC Dynamics",accuracy_data,add_prefix_title)
        if train_bool== True:
            print("Plot Saved Successfully")

    elif selection==2:
        train_bool=traj_plots("Timesteps","Probability Of a Being a Square",f"Performance of Model(Trained without Pc Dynamics) on Cifar 10",results,add_prefix_title)
        if train_bool== True:
            print("Plot Saved Successfully")




















    

