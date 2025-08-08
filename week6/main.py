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
from network import Net as Net
from fwd_train import feedfwd_training
from back_train import feedback_training
from pc_train import class_pc_training
from recon_pc_train import recon_pc_training
from eval_and_plotting import eval_pc_accuracy,recon_pc_loss
import random
import os
import sys
import time
from wb_tracker import init_wandb
import wandb
import argparse
import importlib

start = time.time()

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
     'truck')

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

    trainset = torchvision.datasets.CIFAR10(
    root=datasetpath,
    train=True,
    download=True,
    transform=transform)

    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
    root=datasetpath,
    train=False,
    download=True,
     transform=transform)

    testloader = torch.utils.data.DataLoader(
    testset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


def create_priors_dict(gammaset, betaset, alphaset):

    hyp_dict = {}

    for a, b, c in zip(gammaset, betaset, alphaset):
        key_name = 'Gamma: ' + ' '.join([str(s) for s in a]) + ' \n ' + 'Beta: ' + ' '.join(
            [str(s) for s in b]) + ' \n ' + 'Alpha: ' + ' '.join([str(s) for s in c]) + ' \n '
        dict_value = [a, b, c]
        hyp_dict[key_name] = dict_value

    return hyp_dict


def training_using_reconstruction_and_predicitve_coding(net,save_dir, trainloader, testloader,config):

    
    recon_pc_training(net,trainloader,testloader,"train",config)

    print("Model Save Sucessfully")

    return True


def reconstruction_testing_on_random_network(net,save_dir, trainloader, testloader,config):
    
    criterion=nn.CrossEntropyLoss()
    print("This is a test to see the loss values when the network is not trained at all")
    for i in range(10):
        zp_test_loss=recon_pc_loss(net,trainloader,config)
        print(f"Zp Model Recon Loss for epoch{i}",zp_test_loss)
        pc_test_accuracy,pc_test_loss,pc_test_recon_loss=eval_pc_accuracy(net,trainloader,config,criterion)
        print("My Preditive Coding Model with Dense layers")
        print(f"Accuracy:{pc_test_accuracy} and Recon Loss:{pc_test_recon_loss}")

    return None


def fine_tuning_using_classification(net,save_dir, trainloader, testloader,config):
    
    for iteration_index in range(8):
        print(f"The Iteration{iteration_index}:")
        print("================================")
        net.load_state_dict(torch.load(f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',
        map_location=config.device,weights_only=True))
        train_bool=recon_pc_training(net,trainloader,testloader,"fine_tuning",config)
        if train_bool == True:
            torch.save(net.state_dict(), f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
            print("Model Saved Sucessfully")

    return train_bool


def main():
    init_wandb(config.batch_size,config.epochs,config.lr,config.momentum,config.seed,config.device,config.training_condition,config.timesteps,config.gammaset,config.betaset,config.alphaset,config.datasetpath,config.experiment_name,config.noise_type,config.noise_param,config.model_name)

    save_dir = os.path.join("result_folder", f"Seed_{config.seed}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"Accuracy_Stats_{config.seed}.txt")
    net = Net().to(config.device)
    wandb.watch(net, log="all", log_freq=10)
    trainloader, testloader = train_test_loader(config.datasetpath)

    if config.training_condition == "fine_tuning_classification":
        train_bool = fine_tuning_using_classification(net,save_dir, trainloader, testloader,config)

    if config.training_condition == "random_network_testing":
            train_bool = reconstruction_testing_on_random_network(net,save_dir, trainloader, testloader,config)


    for iteration_index in range(8):
        
        print(f"The Iteration{iteration_index}:")
        print("================================")
        if iteration_index != 0:
            net.load_state_dict(
            torch.load(f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))

        if config.training_condition == "recon_pc_train":
            train_bool = training_using_reconstruction_and_predicitve_coding(net,save_dir, trainloader, testloader,config)
            if train_bool == True:
                torch.save(net.state_dict(), f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
                print("Training Sucessful")
                

    end = time.time()
    diff = end - start
    diff = diff / 60
    wandb.log({"Time Taken to Run the Code(Mins)": diff})

    wandb.finish()


def load_config(config_name):
    return importlib.import_module(config_name)

# This line ensures safe multiprocessing
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Add config directory to Python path
    sys.path.append(os.path.abspath("configs"))
    
    config = load_config(args.config)
    
    main()

