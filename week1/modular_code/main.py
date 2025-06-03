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
from back_train import feedback_training
from config import batch_size,epochs,lr,momentum
import random
import os
import sys

seed=int(sys.argv[1])

def set_seed(seed):
    
    #for random module
    random.seed(seed)
    #for numpy
    np.random.seed(seed)
    #for cpu
    torch.manual_seed(seed)
    #for gpus
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #for reproducibility
    torch.backends.cudnn.deterministic=True
    #disable auto-optimization
    torch.backends.cudnn.benchmark=False


set_seed(seed)

#Normalizing the images
transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


trainset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets',train=True,download=True,transform=transform)

#trainset=torchvision.datasets.CIFAR10(root="D:\datasets",train=True,download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='/home/ajinkya/projects/datasets',train=False,download=True,transform=transform)

#testset=torchvision.datasets.CIFAR10(root="D:\datasets",train=False,download=True,transform=transform)

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
    save_dir=os.path.join("result_folder",f"Seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)
    file_path=os.path.join(save_dir,f"Accuracy_Stats_{seed}.txt")
    with open(file_path,"w") as f:
        f.write(f"Results for hyperparameters settings Epochs={epochs},Batch Size= {batch_size},learning rate= {lr},momentum={momentum} \n")
        f.write(" \n")
        f.write(" \n")
    ft_AB,ft_BC,ft_CD,ft_DE,output=feedfwd_training(net,trainloader,testloader,lr,momentum,save_dir)
    #visualize_model(net)
    feedback_training(net,trainloader,testloader,lr,momentum,save_dir)




# This line ensures safe multiprocessing
if __name__ == "__main__":
    main()
