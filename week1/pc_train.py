import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import evaluation_metric,evaluation_reconstruction,plot_metrics
from config import epochs,seed
import os

def pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha):
    
    #net.eval()

    forward_params = [
    net.conv1, net.conv2, net.fc1, net.fc2, net.fc3]

    for module in forward_params:
        for param in module.parameters():
            param.requires_grad = False

    feedback_params = [
        net.fc3_fb, net.fc2_fb, net.fc1_fb, 
        net.deconv2_fb, net.deconv1_fb
    ]
    for module in feedback_params:
        for param in module.parameters():
            param.requires_grad = False

    

    #criterion = nn.CrossEntropyLoss()
    #gamma=[0.2,0.7]
    #beta=[0.8,0.3]
    #gamma=[0.8,0.3]
    #beta=[0.2,0.7]
    #alpha=[0.3,0.8]
    
    timesteps=4
    
    total_correct=np.zeros(timesteps+1)
    total_samples=0
    
    for batch_id,batch in enumerate(testloader):
        images,labels=batch
        ft_AB,ft_BC,ft_CD,ft_DE,output,indices_AB,indices_BC = net.feedforward_pass(images)
        ft_AB.requires_grad=True
        ft_BC.requires_grad=True
        output,ft_AB_fc_temp,ft_BC_pc_temp,ft_CD_temp,ft_DE_temp=net.predictive_coding_pass(images,ft_AB,ft_BC,ft_CD,ft_DE,beta,gamma,alpha,images.size(0))     
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
    
