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
from PIL import Image
import json
from add_noise import noisy_img

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

def recon_pc_loss(net,dataloader,config):
    
    total_loss=[]
    test_loss=0
    for batch_idx,batch in enumerate(dataloader):
        images,labels=batch
        images,labels=images.to(config.device),labels.to(config.device)

        ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
        ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
        ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
        ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

        ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

        # Re-enable gradients after feedforward_pass overwrites the tensors
        # Only enable gradients for the specific tensors that need them
        ft_AB_pc_temp.requires_grad_(True)
        ft_BC_pc_temp.requires_grad_(True)
        ft_CD_pc_temp.requires_grad_(True)
        ft_DE_pc_temp.requires_grad_(True)

        final_loss=0

        for i in range(config.timesteps):
            #print("Timestep",i)
            #print("Batch Id",batch_idx)
            ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
            final_loss+=loss_of_layers

        final_loss=final_loss/config.timesteps
        total_loss.append(final_loss.item())
        # Clear batch tensors
        del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,loss_of_layers,final_loss
        torch.cuda.empty_cache()

    test_loss=np.mean(total_loss)

    return test_loss


def eval_pc_accuracy(net,dataloader,config,criterion):

    total_correct = np.zeros(config.timesteps + 1)  # ✅ Initialize here
    total_samples = 0  # ✅ Initialize here
    running_loss=[]
    val_recon_loss=[]

    for batch_id,batch in enumerate(dataloader):
        images,labels=batch
        #Adding noise to the image
        #images=noisy_img(images,noise_type,noise_param)
        
        # Move data to the same device as the model
        images, labels = images.to(config.device), labels.to(config.device)
        ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32)
        ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16)
        ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8)
        ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4)

        ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,ft_FG_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

        # Re-enable gradients after feedforward_pass overwrites the tensors
        ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
        ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
        ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
        ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)

        _,predicted=torch.max(output,1)
        total_correct[0]+=(predicted==labels).sum().item()
        final_loss=0
        recon_loss=0
        for i in range(config.timesteps):  
            output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,loss_of_layers=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
            loss=criterion(output,labels)
            final_loss+=loss
            recon_loss+=(loss_of_layers/4.0)
            _,predicted=torch.max(output,1)
            total_correct[i+1]+=(predicted==labels).sum().item()

        total_samples+=labels.size(0)
        final_loss=final_loss/config.timesteps
        recon_loss=recon_loss/config.timesteps
        running_loss.append(final_loss.item())
        val_recon_loss.append(recon_loss.item())

    accuracy=[100 * c /total_samples for c in total_correct]    
    accuracy=np.array(accuracy)
    mean_acc=np.mean(accuracy)
    test_loss=np.mean(running_loss)
    test_recon_loss=np.mean(val_recon_loss)


    return mean_acc,test_loss,test_recon_loss


def eval_pc_ill_accuracy(net, dataloader, config, criterion):

    total_correct = np.zeros(config.timesteps + 1)  # Initialize accuracy buffer
    total_samples = 0
    running_loss = []
    val_recon_loss = []
    
    for images, labels, _ in dataloader:

        # Move to device
        images, labels = images.to(config.device), labels.to(config.device)

        for noise in np.arange(0, 0.35, 0.05):

            images_noisy = noisy_img(images, "gauss", round(noise, 2))

            ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32)
            ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16)
            ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8)
            ft_DE_pc_temp = torch.zeros(config.batch_size, 64, 4, 4)

            ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
            ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                images_noisy,
                ft_AB_pc_temp, ft_BC_pc_temp,
                ft_CD_pc_temp, ft_DE_pc_temp
            )

            # Re-enable grad
            ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
            ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)

            # First-step prediction accuracy
            _, predicted = torch.max(output, 1)
            total_correct[0] += (predicted == labels).sum().item()

            final_loss = 0
            recon_loss = 0

            for i in range(config.timesteps):

                output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(
                    images_noisy,
                    ft_AB_pc_temp, ft_BC_pc_temp,
                    ft_CD_pc_temp, ft_DE_pc_temp,
                    ft_EF_pc_temp,
                    config.betaset, config.gammaset, config.alphaset,
                    images_noisy.size(0)
                )

                loss = criterion(output, labels)
                final_loss += loss
                recon_loss += (loss_of_layers / 4.0)

                _, predicted = torch.max(output, 1)
                total_correct[i + 1] += (predicted == labels).sum().item()

            total_samples += labels.size(0)

            final_loss = final_loss / config.timesteps
            recon_loss = recon_loss / config.timesteps

            running_loss.append(final_loss.item())
            val_recon_loss.append(recon_loss.item())

    # Accuracy per timestep
    accuracy = [100 * c / total_samples for c in total_correct]
    accuracy = np.array(accuracy)

    mean_acc = np.mean(accuracy)
    test_loss = np.mean(running_loss)
    test_recon_loss = np.mean(val_recon_loss)

    return mean_acc, test_loss, test_recon_loss




def recon_loss(net,dataloader,batch_size,device,criterion_recon):
    running_loss = []
   

    for batch_idx, batch in enumerate(dataloader):
        ft_AB = torch.zeros(batch_size, 6, 32, 32)
        ft_BC = torch.zeros(batch_size, 16, 16, 16)
        ft_CD = torch.zeros(batch_size, 32, 8, 8)
        ft_DE = torch.zeros(batch_size,64,4,4)
        images, labels = batch
        images,labels=images.to(device),labels.to(device)
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
   

    return True


def save_training_metrics(metrics_history, save_dir, model_name):
    """Save training metrics to JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"metrics_{model_name}.json")
    
    with open(filepath, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"Metrics saved to: {filepath}")
    return filepath


def plot_training_curves(metrics_history, save_dir, model_name):
    """Plot training and validation curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    # Determine number of subplots needed
    has_accuracy = 'train_acc' in metrics_history and metrics_history['train_acc']
    has_recon = 'train_recon_loss' in metrics_history and metrics_history['train_recon_loss']
    
    if has_recon:
        # 3 subplots: Loss, Accuracy, Reconstruction Loss
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    else:
        # 2 subplots: Loss, Accuracy
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # ========================================
    # SUBPLOT 1: Classification/Task Loss
    # ========================================
    axes[0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    if 'test_loss' in metrics_history and metrics_history['test_loss']:
        axes[0].plot(epochs, metrics_history['test_loss'], 'r-', label='Test Loss', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # ========================================
    # SUBPLOT 2: Accuracy
    # ========================================
    if has_accuracy:
        axes[1].plot(epochs, metrics_history['train_acc'], 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
        if 'test_acc' in metrics_history and metrics_history['test_acc']:
            axes[1].plot(epochs, metrics_history['test_acc'], 'r-', label='Test Acc', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 100])
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[1].text(0.5, 0.5, 'No Accuracy Data\n(Reconstruction Training Only)', 
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].axis('off')
    
    # ========================================
    # SUBPLOT 3: Reconstruction Loss (if available)
    # ========================================
    if has_recon:
        axes[2].plot(epochs, metrics_history['train_recon_loss'], 'g-', label='Train Recon Loss', 
                     linewidth=2, marker='o', markersize=4)
        if 'test_recon_loss' in metrics_history and metrics_history['test_recon_loss']:
            axes[2].plot(epochs, metrics_history['test_recon_loss'], 'orange', label='Test Recon Loss', 
                         linewidth=2, marker='s', markersize=4)
        axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Reconstruction Loss', fontsize=12, fontweight='bold')
        axes[2].set_title('Predictive Coding Reconstruction Loss', fontsize=14, fontweight='bold')
        axes[2].legend(loc='best', fontsize=10)
        axes[2].grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle(f'Training Metrics: {model_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, f"training_curves_{model_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {filepath}")
    return filepath


