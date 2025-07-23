import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from config import seed, device, batch_size, epochs, lr, momentum, timesteps,training_condition
import os
import wandb
from PIL import Image

def evaluation_metric(net,dataloader,batch_size,device):
    # Testing
    #net.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            
            ft_AB = torch.randn(batch_size, 6, 32, 32)
            ft_BC = torch.randn(batch_size, 16, 16, 16)
            ft_CD = torch.randn(batch_size, 32, 8, 8)
            ft_DE = torch.randn(batch_size,64,4,4)
            images, labels = batch
            images,labels=images.to(device),labels.to(device)
            _,_,_,_,_,output = net.feedforward_pass(images,ft_AB,ft_BC,ft_CD,ft_DE)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * (total_correct / total_samples)


    return accuracy

#It's showing the error not the accuracy fix that

def evaluation_of_loss_metric(net,dataloader,batch_size,device,criterion):
    # Testing
    #net.eval()
    final_loss=0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            
            ft_AB = torch.randn(batch_size, 6, 32, 32)
            ft_BC = torch.randn(batch_size, 16, 16, 16)
            ft_CD = torch.randn(batch_size, 32, 8, 8)
            ft_DE = torch.randn(batch_size,64,4,4)
            images, labels = batch
            images,labels=images.to(device),labels.to(device)
            _,_,_,_,_,output = net.feedforward_pass(images,ft_AB,ft_BC,ft_CD,ft_DE)
            loss=criterion(output,labels)
            final_loss+=loss.item()

    final_loss=final_loss/len(dataloader)


    return final_loss


def plot_metrics(x,y,save_dir,xtitle,ytitle,title,savetitle,seed):
    
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
    plt.savefig(file_path,dpi=150,bbox_inches='tight',facecolor='white',edgecolor='none')
    # Log to WandB with proper caption and key
    wandb.log({
        f"plots/{savetitle}": wandb.Image(file_path, caption=f"{title}")
    })
    
    return True

def plot_multiple_metrics(x,y_dict,save_dir,xtitle,ytitle,title,savetitle,seed):

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
    plt.legend(loc='lower left')
    plt.savefig(file_path,dpi=150,bbox_inches='tight',facecolor='white',edgecolor='none')
    # Log to WandB with proper caption and key
    wandb.log({
        f"plots/{savetitle}": wandb.Image(file_path, caption=f"{title}")
    })

    return True

