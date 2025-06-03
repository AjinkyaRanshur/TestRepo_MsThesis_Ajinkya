import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import seed
import os

def evaluation_metric(net,direction,testloader):
    # Testing
    net.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch
            _,_,_,_,output,_,_ = net.feedforward_pass(images)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * (total_correct / total_samples)


    return accuracy

#It's showing the error not the accuracy fix that

def evaluation_reconstruction(net,testloader):
    net.eval()
    total_pixels = 0
    correct_pixels = 0
    threshold=0.1
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch
            ft_AB,ft_BC,ft_CD,ft_DE,output,indices_AB,indices_BC = net.feedforward_pass(images)
            _,_,_,_, xpred = net.feedback_pass(output,indices_AB,indices_BC,ft_AB,ft_BC,ft_CD,ft_DE)
            diff=torch.abs(xpred-images)
            correct=(diff<threshold).float().sum().item()
            total=images.numel()
            correct_pixels+=correct
            total_pixels+=total

    accuracy = 100 * (correct_pixels / total_pixels)

    return accuracy

def plot_metrics(x,y,direction,save_dir):

    if direction=="forward":
        title="Forward Training Evaluation"

    if direction=="backward":
        title="Backward Training Evaluation"
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=2, markersize=6)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.ylim(0,5)
    plt.yticks(range(0,5,1))
    #plt.xticks(x)
    plt.tight_layout()
    os.makedirs(save_dir,exist_ok=True)
    file_path=os.path.join(save_dir,f"avgloss_vs_epoch_{direction}_{seed}.png")
    plt.savefig(file_path)

    return True
