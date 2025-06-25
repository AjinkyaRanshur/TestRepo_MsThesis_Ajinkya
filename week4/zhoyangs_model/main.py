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
from pc_train import pc_training
from eval_and_plotting import plot_multiple_metrics
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
    trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
    root=datasetpath,
    train=False,
    download=True,
     transform=transform)

    testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


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

    main()

