import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import classification_accuracy_metric,classification_loss_metric,plot_metrics,recon_pc_loss,eval_pc_accuracy
import os
import wandb
from wb_tracker import init_wandb
from add_noise import noisy_img
import torchvision.utils as vutils
from PIL import Image

def visualize_dataset_samples(dataloader, num_batches=3, samples_per_batch=8):
    """
    Visualize samples from the custom illusory dataset and log to wandb
    """
    class_names = {0: 'square', 1: 'circle', 2: 'AllOut', 3: 'AllIn'}
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        images, labels = batch
        
        # Select samples to visualize
        num_samples = min(samples_per_batch, images.size(0))
        sample_images = images[:num_samples]
        sample_labels = labels[:num_samples]
        
        # Denormalize images for proper visualization
        # Your normalization: (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        
        # Denormalize
        denorm_images = sample_images * std + mean
        # Clamp to [0, 1] range
        denorm_images = torch.clamp(denorm_images, 0, 1)
        
        # Create grid
        grid = vutils.make_grid(denorm_images, nrow=4, padding=2, normalize=False)
        
        # Create caption with labels
        caption = f"Batch {batch_idx + 1}: " + ", ".join([
            f"{class_names.get(label.item(), f'Class_{label.item()}')}" 
            for label in sample_labels
        ])
        
        # Log to wandb
        wandb.log({
            f"Dataset_Visualization/Batch_{batch_idx + 1}": wandb.Image(
                grid, 
                caption=caption
            )
        })
        
        print(f"Logged batch {batch_idx + 1} with labels: {sample_labels.tolist()}")

def illusion_pc_training(net,trainloader,testloader,train_bool,config):

    if train_bool=="train":
        # First, visualize the training dataset
        print("Visualizing training dataset samples...")
        visualize_dataset_samples(trainloader, num_batches=3, samples_per_batch=8)
        # Also visualize test dataset to see the difference
        print("Visualizing test dataset samples...")
        visualize_dataset_samples(testloader, num_batches=2, samples_per_batch=8)

        criterion=nn.CrossEntropyLoss()
        loss_arr=[]
        optimizer= optim.Adam(list(net.fc1.parameters())+list(net.fc2.parameters())+list(net.fc3.parameters()), lr=config.lr)
        for epoch in range(config.epochs):
            running_loss = []
            net.train()
            for batch_idx, batch in enumerate(trainloader):
                images, labels = batch
                # Log a few samples from the first batch of each epoch for monitoring
                if batch_idx == 0 and epoch % 5 == 0:  # Every 5th epoch
                    # Denormalize for visualization
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
                    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
                    denorm_images = images * std + mean
                    denorm_images = torch.clamp(denorm_images, 0, 1)
            

    if train_bool=="fine_tuning":
        # First, visualize the training dataset

        criterion=nn.CrossEntropyLoss()
        optimizer= optim.Adam(list(net.fc1.parameters())+list(net.fc2.parameters())+list(net.fc3.parameters()), lr=config.lr)

        loss_arr=[]
        ##In zhoyang's paper finetuning was for only 25 epochs
        for epoch in range(config.epochs):
            running_loss=[]
            val_recon_loss=[]
            total_correct = np.zeros(config.timesteps + 1)  # ✅ Initialize here
            total_samples = 0  # ✅ Initialize here
            net.train()
            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(config.device),labels.to(config.device)

                ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
                ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
                ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
                ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,ft_FG_ppc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)
                
                _,predicted=torch.max(output,1)
                total_correct[0]+=(predicted==labels).sum().item()

                # In pc_train.py training loop
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)


                optimizer.zero_grad()
                final_loss=0
                train_recon_loss=0
                for i in range(config.timesteps):
                    output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,loss_of_layers=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
                    loss=criterion(output,labels)
                    final_loss+=loss
                    train_recon_loss+=loss_of_layers
                    _,predicted=torch.max(output,1)
                    total_correct[i+1]+=(predicted==labels).sum().item()

                total_samples+=labels.size(0)
                final_loss=final_loss/config.timesteps
                train_recon_loss=train_recon_loss/config.timesteps
                final_loss.backward()
                optimizer.step()
                running_loss.append(final_loss.item())
                val_recon_loss.append(train_recon_loss.item())

                #Clear Batches
                del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,ft_EF_pc_temp,loss_of_layers
                torch.cuda.empty_cache()
            
            accuracy=[100 * c /total_samples for c in total_correct]
            accuracy=np.array(accuracy)
            train_accuracy=np.mean(accuracy)
            avg_loss=np.mean(running_loss)
            avg_recon_loss=np.mean(val_recon_loss)
            print(f"Epoch:{epoch} and AverageLoss:{avg_loss} and Reconstruction Loss:{train_recon_loss}")
            net.eval()
            test_accuracy,test_loss,test_recon_loss=eval_pc_accuracy(net,testloader,config,criterion)
            metrics={"Fine_Tuning_With_Classification/train_loss":avg_loss,"Fine_Tuning_With_Classification/test_loss":test_loss,"Fine_Tuning_With_Classification/Test_Accuracy":test_accuracy,"Fine_Tuning_With_Classification/Training_accuracy":train_accuracy,"Fine_Tuning_With_Classification/Recon_Training_loss":avg_recon_loss,"Fine_Tuning_With_Classification/Recon_Testing_loss":test_recon_loss }
            wandb.log(metrics)
            loss_arr.append(avg_loss)

        return True


