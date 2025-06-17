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
from config import batch_size,epochs,lr,momentum,seed,device,training_condition,load_model,save_model,timesteps,gammaset,betaset,alphaset,datasetpath,experiment_name
from pc_train import pc_training
from eval_and_plotting import plot_multiple_metrics
import random
import os
import sys
import time
from wb_tracker import init_wandb
import wandb

start=time.time()
classes= ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


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

def train_test_loader(datasetpath):
    #Normalizing the images
    transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    trainset=torchvision.datasets.CIFAR10(root=datasetpath,train=True,download=True,transform=transform)

    trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

    testset=torchvision.datasets.CIFAR10(root=datasetpath,train=False,download=True,transform=transform)

    testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0)

    
    return trainloader,testloader


def create_priors_dict(gammaset,betaset,alphaset):

    hyp_dict={}

    for a,b,c in zip(gammaset,betaset,alphaset):
        key_name='Gamma: ' + ' '.join([str(s) for s in a]) + ' \n ' + 'Beta: ' + ' '.join([str(s) for s in b]) + ' \n ' + 'Alpha: ' + ' '.join([str(s) for s in c]) + ' \n '
        dict_value=[a,b,c]
        hyp_dict[key_name]=dict_value

    return hyp_dict


def training_using_ff_fb(save_dir,trainloader,testloader,net):
    ft_AB,ft_BC,ft_CD,ft_DE,output=feedfwd_training(net,trainloader,testloader,lr,momentum,save_dir)
    feedback_training(net,trainloader,testloader,lr,momentum,save_dir)
    return True

def training_using_predicitve_coding(save_dir,trainloader,testloader,net):
    accuracy_dict={}
    #net = Net().to(device)
    gamma_train=[0.33,0.33,0.33]
    beta_train=[0.33,0.33,0.33]
    alpha_train=[0.01,0.01,0.01]
    pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma_train,beta_train,alpha_train,"train")
    torch.save(net.state_dict(), 'cnn_model.pth')
    print("Model Save Sucessfully")

    return True

def testing_model(save_dir,trainloader,testloader,net):
    #net=Net()
    net.load_state_dict(torch.load('cnn_model.pth', map_location=device, weights_only=True))
    #net = net.to(device)  # Ensure the entire model is on the correct device
    hyp_dict=create_priors_dict(gammaset,betaset,alphaset)
    i=0
    accuracy_dict={}
    iters=range(0,timesteps+1,1)
    for key,value in hyp_dict.items():
        set_seed(seed)
        gamma_i,beta_i,alpha_i=value
        accuracy_i=pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma_i,beta_i,alpha_i,"test")
        i+=1
        accuracy_dict[key]=accuracy_i

    return accuracy_dict

def main():
    init_wandb(experiment_name)
    save_dir=os.path.join("result_folder",f"Seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)
    file_path=os.path.join(save_dir,f"Accuracy_Stats_{seed}.txt")
    net=Net().to(device)
    wandb.watch(net,log="all",log_freq=10)
    trainloader,testloader=train_test_loader(datasetpath)
    iters=range(0,timesteps+1,1)
    if training_condition=="ff_fb_train":
        train_bool=training_using_ff_fb(save_dir,trainloader,testloader,net)
        if train_bool==True:
            print("Training Sucessful")

    if training_condition=="pc_train":
        train_bool=training_using_predicitve_coding(save_dir,trainloader,testloader,net)
        if train_bool==True:
            print("Training Sucessful")

    accuracy_dict=testing_model(save_dir,trainloader,testloader,net)
    #wandb.log(accuracy_dict)
    plot_multiple_metrics(iters,accuracy_dict,save_dir,"Timesteps","Accuracies for different priors","Predicitive Coding Performance for Various Hyperparameter Configrations","pc_multiplehp_accuracy_vs_timesteps")

    end=time.time()
    diff=end - start
    diff=diff/60
    wandb.log({"Time Taken to Run the Code(Mins)":diff})

    wandb.finish()

# This line ensures safe multiprocessing
if __name__ == "__main__":
    main()













