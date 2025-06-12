import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import seed,device
import os

def evaluation_metric(net,direction,testloader):
    # Testing
    #net.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch
            images,labels=images.to(device),labels.to(device)
            _,_,_,_,output = net.feedforward_pass(images)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * (total_correct / total_samples)


    return accuracy

#It's showing the error not the accuracy fix that

def evaluation_reconstruction(net,testloader):
    #net.eval()
    total_pixels = 0
    correct_pixels = 0
    threshold=0.1
    for batch_idx, batch in enumerate(testloader):
        images, labels = batch
        images,labels=images.to(device),labels.to(device)
        ft_AB,ft_BC,ft_CD,ft_DE,output,indices_AB,indices_BC = net.feedforward_pass(images)
        _,_,_,_, xpred = net.feedback_pass(output,indices_AB,indices_BC,ft_AB,ft_BC,ft_CD,ft_DE)
        diff=torch.abs(xpred-images)
        correct=(diff<threshold).float().sum().item()
        total=images.numel()
        correct_pixels+=correct
        total_pixels+=total

    accuracy = 100 * (correct_pixels / total_pixels)

    return accuracy

def plot_metrics(x,y,save_dir,xtitle,ytitle,title,savetitle):

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=2, markersize=6)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    #plt.ylim(0,5)
    #plt.yticks(range(0,5,1))
    plt.xticks(x)
    plt.tight_layout()
    os.makedirs(save_dir,exist_ok=True)
    file_path=os.path.join(save_dir,f"{savetitle}_{seed}.png")
    plt.savefig(file_path)

    return True

def plot_multiple_metrics(x,y_dict,save_dir,xtitle,ytitle,title,savetitle):

    # Plotting
    plt.figure(figsize=(8, 6))
    for key,value in y_dict.items():
        plt.plot(x,value,linewidth=2,markersize=6,label=key)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.xticks(x)
    plt.tight_layout()
    os.makedirs(save_dir,exist_ok=True)
    file_path=os.path.join(save_dir,f"{savetitle}_{seed}.png")
    plt.legend()
    plt.savefig(file_path)

    return True

