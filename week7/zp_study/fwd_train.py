import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import classification_accuracy_metric,classification_loss_metric,plot_metrics
import os
import wandb
from wb_tracker import init_wandb

def feedfwd_training(net,trainloader,testloader,lr,momentum,save_dir,epochs,seed,device,batch_size):
    
    net.train()

    forward_params = [
    net.conv1, net.conv2, net.conv3,net.conv4,net.fc1, net.fc2,net.fc3]

    for module in forward_params:
        for param in module.parameters():
            param.requires_grad = True

    feedback_params = [
        net.fc2_fb, net.fc1_fb,net.deconv4_fb,net.deconv3_fb, 
        net.deconv2_fb, net.deconv1_fb
    ]
    for module in feedback_params:
        for param in module.parameters():
            param.requires_grad = False


    criterion = nn.CrossEntropyLoss()
    optimizer_fwd = optim.SGD(list(net.conv1.parameters())+list(net.conv2.parameters())+list(net.conv3.parameters())+list(net.conv4.parameters())+list(net.fc1.parameters())+list(net.fc2.parameters()), lr=lr, momentum=momentum)
    loss_arr = []
    acc_arr=[]
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            ft_AB = torch.randn(batch_size, 6, 32, 32)
            ft_BC = torch.randn(batch_size, 16, 16, 16)
            ft_CD = torch.randn(batch_size, 32, 8, 8)
            ft_DE = torch.randn(batch_size,64,4,4)
            images, labels = batch
            images,labels=images.to(device),labels.to(device)
            optimizer_fwd.zero_grad()
            ft_AB,ft_BC,ft_CD,ft_DE,ft_EF,ft_FG,ypred = net.feedforward_pass(images,ft_AB,ft_BC,ft_CD,ft_DE)
            loss = criterion(ypred, labels)
            loss.backward()
            optimizer_fwd.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        test_loss=classification_loss_metric(net,testloader,batch_size,device,criterion)
        test_accuracy=classification_accuracy_metric(net,testloader,seed,device)
        train_accuracy=classification_accuracy_metric(net,trainloader,seed,device)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        metrics={"Reconstruction_Model/Classification_train_loss":avg_loss,"Reconstruction_Model/Classification_test_loss":test_loss,"Reconstruction_Model/Classication_train_accuracy":train_accuracy,"Reconstruction_Model/Classification_test_accuracy":test_accuracy}
        wandb.log(metrics)


    print("Forward Training Succesful")

    return ft_AB,ft_BC,ft_CD,ft_DE,ft_EF,ypred


