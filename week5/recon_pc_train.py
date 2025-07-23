import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import evaluation_metric,evaluation_of_loss_metric,plot_metrics
import os
import wandb
from wb_tracker import init_wandb
from add_noise import noisy_img
import torchvision.utils as vutils

def recon_pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha,pc_train_bool,epochs,seed,device,timesteps,batch_size,noise_type,noise_param):

    if pc_train_bool=="train":
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.SGD(net.parameters(),lr=lr,momentum=momentum)
        loss_arr=[]
        for epoch in range(epochs):
            running_loss=[]

            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(device),labels.to(device)
                
                ft_AB_pc_temp = torch.zeros(batch_size, 6, 32, 32).to(device)
                ft_BC_pc_temp = torch.zeros(batch_size, 16, 16, 16).to(device)
                ft_CD_pc_temp = torch.zeros(batch_size, 32, 8, 8).to(device)
                ft_DE_pc_temp = torch.zeros(batch_size,64,4,4).to(device)

                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

                # In pc_train.py training loop       
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)
                ft_EF_pc_temp.requires_grad_(True)
                optimizer.zero_grad()
                final_loss=0
                for i in range(timesteps):
                    output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,loss_of_layers=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,beta,gamma,alpha,images.size(0))
                    loss=criterion(output,labels)
                    final_loss+=loss_of_layers
                    final_loss+=loss
                    final_loss=final_loss/5

                final_loss=final_loss/timesteps
                final_loss.backward()
                optimizer.step()
                running_loss.append(final_loss.item())

            avg_loss=np.mean(running_loss)
            print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
            test_loss=evaluation_of_loss_metric(net,testloader,batch_size,device,criterion)
            test_accuracy=evaluation_metric(net,testloader,seed,device)
            train_accuracy=evaluation_metric(net,trainloader,seed,device)
            metrics={"Reconstruction_with_Predictive_Coding/train_loss":avg_loss,"Reconstruction_with_Predictive_Coding/test_loss":test_loss,"Reconstruction_with_Predictive_Coding/Testing_accuracy":test_accuracy,"Reconstruction_with_Predictive_Coding/Training_accuracy":train_accuracy }
            wandb.log(metrics)
            loss_arr.append(avg_loss)
        iters = range(1, epochs+1)

        return None

