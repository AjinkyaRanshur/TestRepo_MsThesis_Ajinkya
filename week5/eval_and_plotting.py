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

def classification_accuracy_metric(net,dataloader,batch_size,device):
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


def classification_loss_metric(net,dataloader,batch_size,device,criterion):

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

def recon_pc_loss(net,dataloader,batch_size,beta,gamma,alpha,device,criterion,timesteps):
    
    total_loss=0
    for batch_idx,batch in enumerate(dataloader):
        images,labels=batch
        images,labels=images.to(device),labels.to(device)

        ft_AB_pc_temp = torch.zeros(batch_size, 6, 32, 32).to(device)
        ft_BC_pc_temp = torch.zeros(batch_size, 16, 16, 16).to(device)
        ft_CD_pc_temp = torch.zeros(batch_size, 32, 8, 8).to(device)
        ft_DE_pc_temp = torch.zeros(batch_size,64,4,4).to(device)

        ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

        # Re-enable gradients after feedforward_pass overwrites the tensors
        ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
        ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
        ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
        ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)
        ft_EF_pc_temp = ft_EF_pc_temp.requires_grad_(True)

        final_loss=0

        for i in range(timesteps):
            ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,beta,gamma,alpha,images.size(0))
            final_loss+=loss_of_layers

        final_loss=final_loss/timesteps
        total_loss+=final_loss
        # Clear batch tensors
        del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp
        torch.cuda.empty_cache()

    total_loss=total_loss/len(dataloader)

    return total_loss


def eval_pc_accuracy(net,dataloader,batch_size,beta,gamma,alpha,noise_type,noise_param,device,timesteps):

    for batch_id,batch in enumerate(dataloader):
        images,labels=batch
        #Adding noise to the image
        #images=noisy_img(images,noise_type,noise_param)
        
        # Move data to the same device as the model
        images, labels = images.to(device), labels.to(device)
        ft_AB_pc_temp = torch.zeros(batch_size, 6, 32, 32)
        ft_BC_pc_temp = torch.zeros(batch_size, 16, 16, 16)
        ft_CD_pc_temp = torch.zeros(batch_size, 32, 8, 8)
        ft_DE_pc_temp = torch.zeros(batch_size,64,4,4)

        ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

        # Re-enable gradients after feedforward_pass overwrites the tensors
        ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
        ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
        ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
        ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)
        ft_EF_pc_temp = ft_EF_pc_temp.requires_grad_(True)

        _,predicted=torch.max(output,1)
        total_correct[0]+=(predicted==labels).sum().item()

        for i in range(timesteps):
            
            output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,_=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,beta,gamma,alpha,images.size(0))
            _,predicted=torch.max(output,1)
            total_correct[i+1]+=(predicted==labels).sum().item()

        total_samples+=labels.size(0)

    accuracy=[100 * c /total_samples for c in total_correct]    
    accuracy=np.array(accuracy)
    mean_acc=np.mean(accuracy)

    return mean_acc


def recon_loss(net,dataloader,batch_size,device,criterion):
    running_loss = []
    
    for batch_idx, batch in enumerate(dataloader):
        ft_AB = torch.zeros(batch_size, 6, 32, 32)
        ft_BC = torch.zeros(batch_size, 16, 16, 16)
        ft_CD = torch.zeros(batch_size, 32, 8, 8)
        ft_DE = torch.zeros(batch_size,64,4,4)
        images, labels = batch
        images,labels=images.to(device),labels.to(device)
        optimizer_bck.zero_grad()
        ft_AB,ft_BC,ft_CD,ft_DE,ft_EF,output=net.feedforward_pass(images,ft_AB,ft_BC,ft_CD,ft_DE)
        ft_BA,ft_CB,ft_DC,ft_ED,ft_FE,xpred = net.feedback_pass(output,ft_AB,ft_BC,ft_CD,ft_DE,ft_EF)

        # # Flatten ft_BC for comparison with ft_CB (which is also flattened in feedback)
        lossAtoB = criterion_recon(ft_AB, ft_BA)
        lossBtoC = criterion_recon(ft_BC, ft_CB)
        lossCtoD = criterion_recon(ft_CD,ft_DC)
        lossDtoE = criterion_recon(ft_DE,ft_ED)
        lossEtoF = criterion_recon(ft_EF,ft_FE)
        loss_input_and_recon = criterion_recon(xpred, images)
        final_loss=lossAtoB+lossBtoC+lossCtoD+lossDtoE+loss_input_and_recon+lossEtoF
        final_loss=final_loss/6.0
        running_loss.append(final_loss.item())
    avg_loss = np.mean(running_loss)
    
    return avg_loss




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

