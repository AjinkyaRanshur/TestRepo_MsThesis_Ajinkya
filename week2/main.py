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
from config import batch_size,epochs,lr,momentum,seed,device,training_condition,load_model,save_model,hyp_dict
from pc_train import pc_training
from eval_and_plotting import plot_multiple_metrics
import random
import os
import sys
import time

start=time.time()
#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device:{device}")

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


trainset=torchvision.datasets.CIFAR10(root='/exports/home/ajinkya/projects/datasets',train=True,download=True,transform=transform)

#trainset=torchvision.datasets.CIFAR10(root="D:\datasets",train=True,download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='/exports/home/ajinkya/projects/datasets',train=False,download=True,transform=transform)

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
    save_dir=os.path.join("result_folder",f"Seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)
    file_path=os.path.join(save_dir,f"Accuracy_Stats_{seed}.txt")
    with open(file_path,"w") as f:
        f.write(f"Results for hyperparameters settings Epochs={epochs},Batch Size= {batch_size},learning rate= {lr},momentum={momentum} \n")
        f.write(" \n")

    if training_condition=="ff_fb_train":
        net = Net().to(device)
        ft_AB,ft_BC,ft_CD,ft_DE,output=feedfwd_training(net,trainloader,testloader,lr,momentum,save_dir)
        #visualize_model(net)
        feedback_training(net,trainloader,testloader,lr,momentum,save_dir)
        hyp_dict={'Gamma= 0.4,0.2,0.8\n Beta=0.5,0.3,0.2\n alpha=0.01,0.01,0.01\n ':[[0.4,0.2,0.8],[0.5,0.3,0.2],[0.01,0.01,0.01]],'Gamma= 0.2,0.2,0.2\n Beta=0.2,0.4,0.5\n alpha=0.01,0.01,0.01\n ':[[0.2,0.2,0.2],[0.2,0.4,0.5],[0.01,0.01,0.01]],'Gamma= 0.5,0.5,0.5\n Beta=0.5,0.5,0.5\n alpha=0.01,0.01,0.01\n ':[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.01,0.01,0.01]]}
        i=0
        accuracy_dict={}
        iters=range(0,5,1)
        for key,value in hyp_dict.items():
        
            set_seed(seed)
            gamma_i,beta_i,alpha_i=value
            accuracy_i=pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma_i,beta_i,alpha_i,"test")
            i+=1
            accuracy_dict[key]=accuracy_i

        plot_multiple_metrics(iters,accuracy_dict,save_dir,"Timesteps","Accuracies for different priors","Predicitive Coding Performance for Various Hyperparameter Configrations","pc_multiplehp_accuracy_vs_timesteps")

    if training_condition=="pc_train":
        if save_model==True:
            net = Net().to(device)
            gamma_train=[0.3,0.3,0.3]
            beta_train=[0.3,0.3,0.3]
            alpha_train=[0.01,0.01,0.01]
            pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma_train,beta_train,alpha_train,"train")
            torch.save(cnn_model.state_dict(), 'cnn_model.pth')

        if load_model==True:
            net=Net()
            net.load_state_dict(torch.load('cnn_model.pth'))
            #hyp_dict={'Gamma= 0.4,0.2,0.8\n Beta=0.5,0.3,0.2\n alpha=0.01,0.01,0.01\n ':[[0.4,0.2,0.8],[0.5,0.3,0.2],[0.01,0.01,0.01]],'Gamma= 0.2,0.2,0.2\n Beta=0.2,0.4,0.5\n alpha=0.01,0.01,0.01\n ':[[0.2,0.2,0.2],[0.2,0.4,0.5],[0.01,0.01,0.01]],'Gamma= 0.5,0.5,0.5\n Beta=0.5,0.5,0.5\n alpha=0.01,0.01,0.01\n ':[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.01,0.01,0.01]]}
            
            i=0
            accuracy_dict={}
            iters=range(0,5,1)
            for key,value in hyp_dict.items():
                set_seed(seed)
                gamma_i,beta_i,alpha_i=value
                accuracy_i=pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma_i,beta_i,alpha_i,"test")
                i+=1
                accuracy_dict[key]=accuracy_i

            plot_multiple_metrics(iters,accuracy_dict,save_dir,"Timesteps","Accuracies for different priors","Predicitive Coding Performance for Various Hyperparameter Configrations","pc_multiplehp_accuracy_vs_timesteps")

    end=time.time()
    with open(file_path,"a") as f:
        diff=end - start
        diff=diff/60
        f.write(f"Time taken to Run the code {diff} minutes")

# This line ensures safe multiprocessing
if __name__ == "__main__":
    main()
