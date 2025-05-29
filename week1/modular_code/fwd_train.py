import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import evaluation_metric,evaluation_reconstruction,plot_metrics
from config import epochs

def feedfwd_training(net,trainloader,testloader):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer_fwd = optim.SGD(list(net.conv1.parameters())+list(net.conv2.parameters())+list(net.fc1.parameters())+list(net.fc2.parameters())+list(net.fc3.parameters()), lr=0.001, momentum=0.9)
    loss_arr = []
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_fwd.zero_grad()
            ft_AB,ft_BC,ft_CD,ft_DE,ypred = net.feedforward_pass(images)
            loss = criterion(ypred, labels)
            loss.backward()
            optimizer_fwd.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_metric(net,"forward",testloader)
    iters = range(1, epochs+1)
    plot_bool=plot_metrics(iters,loss_arr,"forward")
    if plot_bool==True:
        print("Plots Successfully Stored")
    print(f'Accuracy = {accuracy:.2f}%')

    print("Forward Training Succesful")

    return ft_AB,ft_BC,ft_CD,ft_DE,ypred
