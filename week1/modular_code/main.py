#Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import Net as Net
from fwd_train import feedfwd_training
from config import batch_size

#Normalizing the images
transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


trainset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets',train=True,download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets',train=False,download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0)

classes= ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')




#def visualize_model(net):
#    writer = SummaryWriter('runs/cifar10_experiment')
#    sample_input = torch.randn(1, 3, 32, 32)
#    writer.add_graph(net, sample_input)
#    writer.close()
#    #To launch tensorboard use this command: tensorboard --logdir=runs and then click on the## link that it generates  


def main():
    # Your training and testing code goes here
    net = Net()
    
    ft_AB,ft_BC,ft_CD,ft_DE,output=feedfwd_training(net,trainloader,testloader)
    #visualize_model(net)
    #feedback_training(net,ft_AB,ft_BC,ft_CD,ft_DE,output)




# This line ensures safe multiprocessing
if __name__ == "__main__":
    main()
