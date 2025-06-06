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

def feedfwd_training(net,trainloader,testloader,lr,momentum,save_dir):
    
    net.train()

    forward_params = [
    net.conv1, net.conv2, net.fc1, net.fc2, net.fc3]

    for module in forward_params:
        for param in module.parameters():
            param.requires_grad = True

    feedback_params = [
        net.fc3_fb, net.fc2_fb, net.fc1_fb, 
        net.deconv2_fb, net.deconv1_fb
    ]
    for module in feedback_params:
        for param in module.parameters():
            param.requires_grad = False

    

    criterion = nn.CrossEntropyLoss()
    optimizer_fwd = optim.SGD(list(net.conv1.parameters())+list(net.conv2.parameters())+list(net.fc1.parameters())+list(net.fc2.parameters())+list(net.fc3.parameters()), lr=lr, momentum=momentum)
    loss_arr = []
    acc_arr=[]
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_fwd.zero_grad()
            ft_AB,ft_BC,ft_CD,ft_DE,ypred,_,_ = net.feedforward_pass(images)
            loss = criterion(ypred, labels)
            loss.backward()
            optimizer_fwd.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)
        accuracy=evaluation_metric(net,"forward",testloader)
        acc_arr.append(accuracy)

    iters = range(1, epochs+1)
    plot_bool=plot_metrics(iters,loss_arr,save_dir,"Number of Epochs","Average Loss","Forward Training Loss","AverageLoss_Vs_Epoch_forward")
    plot_bool=plot_metrics(iters,acc_arr,save_dir,"Number of Epochs","Accuracy","Forward Testing Performance","Accuracy_Vs_Epoch_forward")
    if plot_bool==True:
        print("Plots Successfully Stored")
    #file_path=os.path.join(save_dir,f"Accuracy_Stats_{seed}.txt")
    #with open(file_path,"a") as f:
    #    f.write(f"Forward Connection Accuracy= {acc_arr:.2f} with seed = {seed}\n")
    #print(f'Accuracy = {accuracy:.2f}%')

    print("Forward Training Succesful")

    return ft_AB,ft_BC,ft_CD,ft_DE,ypred
