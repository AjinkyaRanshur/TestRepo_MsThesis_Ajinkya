import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from eval_and_plotting import evaluation_metric,evaluation_reconstruction,plot_metrics
import os
<<<<<<< HEAD
#import wandb
#from wb_tracker import init_wandb
=======
import wandb
from wb_tracker import init_wandb
>>>>>>> feac1f8acbfc652eba97239f4f7e66756e8d96b7
#from add_noise import noisy_img
import torchvision.utils as vutils

def pre_training(net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha,epochs,seed,device,timesteps,batch_size,noise_type,noise_param):

    criterion=nn.CrossEntropyLoss()
<<<<<<< HEAD
    optimizer = optim.SGD(
    list(net.encoder1.parameters()) +
    list(net.encoder2.parameters()) +
    list(net.encoder3.parameters()) +
    list(net.decoder1.parameters()) +
    list(net.decoder2.parameters()) +
    list(net.decoder3.parameters()),
    lr=lr
    )
=======
    optimizer=optim.SGD(list(net.encoder1.parameters()),list(net.encoder2.parameters()),list(net.encoder3.parameters()),list(net.decoder1.parameters()),list(net.decoder2.parameters()),list(net.decoder3.parameters()) ,lr=lr)
>>>>>>> feac1f8acbfc652eba97239f4f7e66756e8d96b7
    loss_arr=[]

    for epoch in range(epochs):
        running_loss=[]
        for batch_idx,batch in enumerate(trainloader):
            images,labels=batch
            images,labels=images.to(device),labels.to(device)
            ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,relu_ft_CD = net.feedforward_pass(images)
            # In pc_train.py training loop
                
            ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp.requires_grad_(True)

            optimizer.zero_grad()
            final_loss=0
            for i in range(timesteps):
                output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_CD_pc_temp_relu,layer_loss=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,beta,gamma,alpha,images.size(0))
                loss=criterion(output,labels)
                loss+=layer_loss
                loss=loss/4
                final_loss+=loss

            final_loss=final_loss/timesteps
            final_loss.backward()
            optimizer.step()
            running_loss.append(final_loss.item())

        avg_loss=np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        accuracy=evaluation_metric(net,testloader,seed,device)
        metrics={"PreTraining/train_loss":avg_loss,"PreTraining/testing_accuracy":accuracy,"PreTraining/chance_level":10.00,"PreTraining/Startline":0.00}
<<<<<<< HEAD
        #wandb.log(metrics)
=======
        wandb.log(metrics)
>>>>>>> feac1f8acbfc652eba97239f4f7e66756e8d96b7
        loss_arr.append(avg_loss)
        iters = range(1, epochs+1)

        return True

def fine_tuning(net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha,pc_train_bool,epochs,seed,device,timesteps,batch_size,noise_type,noise_param):

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(list(net.encoder1.parameters()),list(net.encoder2.parameters()),list(net.encoder3.parameters()),list(net.decoder1.parameters()),list(net.decoder2.parameters()),list(net.decoder3.parameters()) ,lr=lr)
    loss_arr=[]

    for epoch in range(epochs):
        running_loss=[]
        for batch_idx,batch in enumerate(trainloader):
            images,labels=batch
            images,labels=images.to(device),labels.to(device)
            ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,relu_ft_CD = net.feedforward_pass(images)
            # In pc_train.py training loop
                
            ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp.requires_grad_(True)

            optimizer.zero_grad()
            final_loss=0
            for i in range(timesteps):
                output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,layer_loss=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,beta,gamma,alpha,images.size(0))
                loss=criterion(output,labels)
                loss+=layer_loss
                loss=loss/4
                final_loss+=loss

            final_loss=final_loss/timesteps
            final_loss.backward()
            optimizer.step()
            running_loss.append(final_loss.item())

        avg_loss=np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        accuracy=evaluation_metric(net,testloader,seed,device)
        metrics={"PreTraining/train_loss":avg_loss,"PreTraining/testing_accuracy":accuracy,"PreTraining/chance_level":10.00}
        #wandb.log(metrics)
        loss_arr.append(avg_loss)
        iters = range(1, epochs+1)

        return None




