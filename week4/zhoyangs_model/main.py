# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train import pre_training
from network import ZP_PC_Net as Net
import random
import os
import sys
import time
#from wb_tracker import init_wandb
#import wandb
import argparse
import importlib

start = time.time()


def set_seed(seed):

    # for random module
    random.seed(seed)
    # for numpy
    np.random.seed(seed)
    # for cpu
    torch.manual_seed(seed)
    # for gpus
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    # disable auto-optimization
    torch.backends.cudnn.benchmark = False

def train_test_loader(datasetpath):
    # Normalizing the images
    #Andrea's nromalization is different figure out why

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    trainset = torchvision.datasets.CIFAR100(
    root=datasetpath,
    train=True,
    download=True,
     transform=transform)

    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
    root=datasetpath,
    train=False,
    download=True,
     transform=transform)

    testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


def create_priors_dict(gammaset, betaset, alphaset):

    hyp_dict = {}

    for a, b, c in zip(gammaset, betaset, alphaset):
        key_name = 'Gamma: ' + ' '.join([str(s) for s in a]) + ' \n ' + 'Beta: ' + ' '.join(
            [str(s) for s in b]) + ' \n ' + 'Alpha: ' + ' '.join([str(s) for s in c]) + ' \n '
        dict_value = [a, b, c]
        hyp_dict[key_name] = dict_value

    return hyp_dict


def train_network(training_condition,net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha,epochs,seed,device,timesteps,batch_size,noise_type,noise_param):

    if training_condition=="pre_training":
        train_bool=pre_training(net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha,epochs,seed,device,timesteps,batch_size,noise_type,noise_param)

    return train_bool


def load_config(config_name):
    return importlib.import_module(config_name)

def main():
    #init_wandb(batch_size, epochs, lr, momentum, seed, device, training_condition, load_model, save_model, timesteps, gammaset, betaset, alphaset, datasetpath,experiment_name,noise_type,noise_param,model_name)

    save_dir = os.path.join("result_folder", f"Seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"Accuracy_Stats_{seed}.txt")
    net = Net().to(device)
    #wandb.watch(net, log="all", log_freq=10)
    trainloader, testloader = train_test_loader(datasetpath)

    #Training the model using these hyperparameters
    gamma_train = [0.33, 0.33, 0.33]
    beta_train = [0.33, 0.33, 0.33]
    alpha_train = [0.01, 0.01, 0.01]

    if training_condition == "pre_training":
        train_bool = train_network(training_condition,net,trainloader,testloader,lr,momentum,save_dir,gamma_train,beta_train,alpha_train,epochs,seed,device,timesteps,batch_size,noise_type,noise_param)
        if train_bool == True:
            torch.save(net.state_dict(), f'{model_name}.pth')
            print("Training Sucessful")

    #accuracy_dict = testing_model(save_dir, trainloader, testloader, net,epochs,seed,device,timesteps,batch_size,noise_type,noise_param)

    end = time.time()
    diff = end - start
    diff = diff / 60
    #wandb.log({"Time Taken to Run the Code(Mins)": diff})

    #wandb.finish()



# This line ensures safe multiprocessing
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Add config directory to Python path
    sys.path.append(os.path.abspath("configs"))

    config = load_config(args.config)

    batch_size=config.batch_size
    epochs=config.epochs
    lr=config.lr
    momentum=config.momentum
    seed=config.seed
    device=config.device
    training_condition=config.training_condition
    load_model=config.load_model
    save_model=config.save_model
    timesteps=config.timesteps
    gammaset=config.gammaset
    betaset=config.betaset
    alphaset=config.alphaset
    datasetpath=config.datasetpath
    experiment_name=config.experiment_name
    model_name=config.model_name
    noise_type=config.noise_type
    noise_param=config.noise_param

    main()

