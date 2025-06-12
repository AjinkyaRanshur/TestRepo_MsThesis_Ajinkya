import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import evaluation_metric,evaluation_reconstruction,plot_metrics
from config import epochs,seed,device,timesteps
import os

def pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha,pc_train_bool):
    
    if pc_train_bool=="train":
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.SGD(net.parameters(),lr=lr,momentum=momentum)
        loss_arr=[]
        for epoch in range(epochs):
            running_loss=[]

            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(device),labels.to(device)
                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_temp,ft_DE_temp,output = net.feedforward_pass(images)
                # In pc_train.py training loop
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_temp.requires_grad_(True)
                ft_DE_temp.requires_grad_(True)
                optimizer.zero_grad()
                final_loss=0
                for i in range(timesteps):
                    output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_temp,ft_DE_temp=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_temp,ft_DE_temp,beta,gamma,alpha,images.size(0))
                    loss=criterion(output,labels)
                    final_loss+=loss

                final_loss=final_loss/timesteps
                final_loss.backward()
                optimizer.step()
                running_loss.append(final_loss.item())

            avg_loss=np.mean(running_loss)
            print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
            loss_arr.append(avg_loss)

        return None



    if pc_train_bool=="test":

        forward_params = [
        net.conv1, net.conv2, net.fc1, net.fc2, net.conv3]

        for module in forward_params:
            for param in module.parameters():
                param.requires_grad = False

        feedback_params = [
            net.deconv3_fb, net.fc2_fb, net.fc1_fb, 
            net.deconv2_fb, net.deconv1_fb
        ]
        for module in feedback_params:
            for param in module.parameters():
                param.requires_grad = False

        
        total_correct=np.zeros(timesteps+1)
        total_samples=0
        
        for batch_id,batch in enumerate(testloader):
            images,labels=batch
            # Move data to the same device as the model
            images, labels = images.to(device), labels.to(device)
            ft_AB,ft_BC,ft_CD,ft_DE,output = net.feedforward_pass(images)
            output,ft_AB_fc_temp,ft_BC_pc_temp,ft_CD_temp,ft_DE_temp=net.predictive_coding_pass(images,ft_AB,ft_BC,ft_CD,ft_DE,beta,gamma,alpha,images.size(0))
            ft_AB_fc_temp.requires_grad=True
            ft_BC_fc_temp.requires_grad=True
            ft_CD_fc_temp.requires_grad=True
            _,predicted=torch.max(output,1)
            total_correct[0]+=(predicted==labels).sum().item()

            for i in range(timesteps):
                output,ft_AB_fc_temp,ft_BC_pc_temp,ft_CD_temp,ft_DE_temp=net.predictive_coding_pass(images,ft_AB_fc_temp,ft_BC_pc_temp,ft_CD_temp,ft_DE_temp,beta,gamma,alpha,images.size(0))
                _,predicted=torch.max(output,1)
                total_correct[i+1]+=(predicted==labels).sum().item()


            total_samples+=labels.size(0)


        accuracy=[100 * c /total_samples for c in total_correct]
        #iters=range(0,timesteps+1)
        #plot_bool=plot_metrics(iters,accuracy,save_dir,"Timesteps","Accuracy","Predictive Coding Performance","pc_timesteps_vs_accuracy")
        print("Accuracy at each timestep:")
        for i, acc in enumerate(accuracy):
            print(f"Timestep {i}: {acc:.2f}%")

        return accuracy
    
