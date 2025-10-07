# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,random_split,DataLoader
from torch.utils.tensorboard import SummaryWriter
from network import Net as Net
from fwd_train import feedfwd_training
from back_train import feedback_training
from pc_train import class_pc_training
from recon_pc_train import recon_pc_training
from eval_and_plotting import eval_pc_accuracy,recon_pc_loss
from illusion_pc_train import illusion_pc_training
from customdataset import SquareDataset
import random
import os
import sys
import time
from wb_tracker import init_wandb
import wandb
import argparse
import importlib
from PIL import Image
import torchvision.utils as vutils
import pandas as pd

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




def train_test_loader(datasetpath,illusion_bool):
    # Normalizing the images
    #Andrea's nromalization is different figure out why    
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
    if illusion_bool == True:
        DATA_DIR = datasetpath
        full_trainset=SquareDataset(os.path.join(DATA_DIR, "metadata.csv"), DATA_DIR,classes_for_use=["Square", "Random"],transform=transform)
        validation_set=SquareDataset(os.path.join(DATA_DIR, "metadata.csv"), DATA_DIR,classes_for_use=["Square", "Random", "All-in", "All-out"],transform=transform)
        train_size=int(0.7 * len(full_trainset))
        test_size=len(full_trainset) - train_size
        generator = torch.Generator().manual_seed(42)  # fixed seed for reproducibility
        train_subset, test_subset = random_split(full_trainset, [train_size, test_size], generator=generator)
        trainloader=DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
        testloader=DataLoader(test_subset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
        validationloader=DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)

    else:
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
    
        validationloader=0
    return trainloader, testloader,validationloader


def training_using_reconstruction_and_predicitve_coding(save_dir, trainloader, testloader,config,iteration_index):
    net = Net(num_classes=10).to(config.device)
    if iteration_index != 0:
        checkpoint_path = f"{config.load_model_path}/{config.model_name}_{iteration_index}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=config.device,weights_only=True)
        net.conv1.load_state_dict(checkpoint["conv1"])
        net.conv2.load_state_dict(checkpoint["conv2"])
        net.conv3.load_state_dict(checkpoint["conv3"])
        net.conv4.load_state_dict(checkpoint["conv4"])
        net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
        net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
        net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
        net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])

    recon_pc_training(net,trainloader,testloader,"train",config)

    # Save only conv layers
    save_path = f'{config.save_model_path}/{config.model_name}_{iteration_index + 1}.pth'
    torch.save({
        "conv1": net.conv1.state_dict(),
        "conv2": net.conv2.state_dict(),
        "conv3": net.conv3.state_dict(),
        "conv4": net.conv4.state_dict(),
        "deconv1_fb": net.deconv1_fb.state_dict(),
        "deconv2_fb": net.deconv2_fb.state_dict(),
        "deconv3_fb": net.deconv3_fb.state_dict(),
        "deconv4_fb": net.deconv4_fb.state_dict(),
    }, save_path)

    print(f"Model Saved Successfully to: {save_path}")

    print("Model Save Sucessfully")

    return True


def fine_tuning_using_classification(save_dir, trainloader, testloader,config,iteration_index):
    
    net = Net(num_classes=10).to(config.device)
    if iteration_index == 0:
        checkpoint_path = f"{config.load_model_path}/{config.model_name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=config.device,weights_only=True)

        net.conv1.load_state_dict(checkpoint["conv1"])
        net.conv2.load_state_dict(checkpoint["conv2"])
        net.conv3.load_state_dict(checkpoint["conv3"])
        net.conv4.load_state_dict(checkpoint["conv4"])
        net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
        net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
        net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
        net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])

    else:
        net.load_state_dict(torch.load(f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))

    train_bool=recon_pc_training(net,trainloader,testloader,"fine_tuning",config)

    if train_bool == True:
        torch.save(net.state_dict(), f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
        print("Model Saved Sucessfully")

    return train_bool


def fine_tuning_using_illusions(save_dir, trainloader, testloader,config,iteration_index):

    net = Net(num_classes=2).to(config.device)

    if iteration_index == 0:
        checkpoint_path = f"{config.load_model_path}/{config.model_name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=config.device,weights_only=True)
        net.conv1.load_state_dict(checkpoint["conv1"])
        net.conv2.load_state_dict(checkpoint["conv2"])
        net.conv3.load_state_dict(checkpoint["conv3"])
        net.conv4.load_state_dict(checkpoint["conv4"])
        net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
        net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
        net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
        net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])

    else:
        net.load_state_dict(torch.load(f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))

    train_bool=illusion_pc_training(net,trainloader,testloader,"fine_tuning",config)

    if train_bool == True:
        torch.save(net.state_dict(), f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
        print("Model Saved Sucessfully")


    return train_bool



def decide_training_model(condition,save_dir, trainloader, testloader, config,iteration_index):
    cond_to_func={
            "recon_pc_train":lambda: training_using_reconstruction_and_predicitve_coding(save_dir, trainloader, testloader, config,iteration_index),
            "fine_tuning_classification": lambda:fine_tuning_using_classification(save_dir, trainloader, testloader,config,iteration_index),
            "random_network_testing": lambda:reconstruction_testing_on_random_network(save_dir, trainloader, testloader,config,iteration_index),
            "illusion_train": lambda:fine_tuning_using_illusions(save_dir, trainloader, testloader,config,iteration_index),
            "illusion_train": lambda:fine_tuning_using_illusions(save_dir, trainloader, testloader,config,iteration_index),
            "recon_comparison": lambda:recon_vs_original(testloader, config, n_images=8, iteration_index=15)}

    result=cond_to_func[condition]()

    return result

def cifar_testing(trainloader,testloader,config,iteration_index):
    net = Net(num_classes=10).to(config.device)
    net.load_state_dict(torch.load(f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))
    class_pc_training(net,trainloader,testloader,"test",config,iteration_index)
    return None

def illusion_testing():

    return None


def main():
    init_wandb(config.batch_size,config.epochs,config.lr,config.momentum,config.seed,config.device,config.training_condition,config.timesteps,config.gammaset,config.betaset,config.alphaset,config.datasetpath,config.experiment_name,config.noise_type,config.noise_param,config.model_name)
    save_dir = os.path.join("result_folder", f"Seed_{config.seed}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"Accuracy_Stats_{config.seed}.txt")

    trainloader, testloader,validationloader= train_test_loader(config.datasetpath,config.illusion_dataset_bool)
    for iteration_index in range(config.iterations):
        if config.training_condition == None:
            break
        print(f"The Iteration{iteration_index}:")
        print("================================")
        decide_training_model(config.training_condition, save_dir, trainloader, testloader, config,iteration_index)

    if config.illusion_dataset_bool == True:
        print("random")
        
    else:
        cifar_testing(trainloader,testloader,config,15)
        
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













