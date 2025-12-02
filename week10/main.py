# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,random_split,DataLoader,Subset
from torch.utils.tensorboard import SummaryWriter
from network import Net as Net
from pc_train import class_pc_training
from recon_pc_train import recon_pc_training
from eval_and_plotting import eval_pc_accuracy,recon_pc_loss
from illusion_pc_train import illusion_pc_training
from custom_ill_train import illusion_pc_training_custom
from customdataset import SquareDataset
import random
import os
import sys
import time
import argparse
import importlib
from PIL import Image
import torchvision.utils as vutils
import pandas as pd
from sklearn.model_selection import train_test_split

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

BASE_RESULTS_DIR = "result_folder"
TRAINING_DIR = os.path.join(BASE_RESULTS_DIR, "training_plots")
os.makedirs(TRAINING_DIR, exist_ok=True)


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


def train_test_loader(datasetpath,illusion_bool,config):
    # Normalizing the images
    #Andrea's nromalization is different figure out why    
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
    if illusion_bool == True:
        DATA_DIR = datasetpath
        full_trainset=SquareDataset(os.path.join(DATA_DIR, "metadata.csv"), DATA_DIR,classes_for_use=["Square", "Random"],transform=transform)
        validation_set=SquareDataset(os.path.join(DATA_DIR, "metadata.csv"), DATA_DIR,classes_for_use=["Square", "Random", "All-in", "All-out"],transform=transform)
        labels=np.array([full_trainset[i][1] for i in range(len(full_trainset))])

        # Stratified split (70% train, 30% test)
        train_idx, test_idx = train_test_split(
            np.arange(len(full_trainset)),
            test_size=0.3,
            random_state=42,
            stratify=labels
        )

        # Subset creation
        train_subset = Subset(full_trainset, train_idx)
        test_subset = Subset(full_trainset, test_idx)

        # DataLoaders
        trainloader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
        testloader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
        validationloader = DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)

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


def training_using_reconstruction_and_predicitve_coding(save_dir, trainloader, testloader,config,iteration_index,metrics_history):
    net = Net(num_classes=10).to(config.device)
    if iteration_index != 0:
        checkpoint_path = f"{config.load_model_path}/recon_models/{config.model_name}_{iteration_index}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=config.device,weights_only=True)
        net.conv1.load_state_dict(checkpoint["conv1"])
        net.conv2.load_state_dict(checkpoint["conv2"])
        net.conv3.load_state_dict(checkpoint["conv3"])
        net.conv4.load_state_dict(checkpoint["conv4"])
        net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
        net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
        net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
        net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])

    metrics_history=recon_pc_training(net,trainloader,testloader,"train",config,metrics_history)

    # Save only conv layers
    save_path = f'{config.save_model_path}/recon_models/{config.model_name}_{iteration_index + 1}.pth'
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

    return metrics_history

def reconstruction_testing_on_random_network(net,save_dir, trainloader, testloader,config,metrics_history):
    

    return None


def fine_tuning_using_classification(save_dir, trainloader, testloader,config,iteration_index,metrics_history):
    
    net = Net(num_classes=10).to(config.device)
    if iteration_index == 0:
        checkpoint_path = f"{config.load_model_path}/recon_models/{config.model_name}_15.pth"
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
        net.load_state_dict(torch.load(f'{config.load_model_path}/cifar10_models/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))

    metrics_history=recon_pc_training(net,trainloader,testloader,"fine_tuning",config,metrics_history)

    if train_bool == True:
        torch.save(net.state_dict(), f'{config.save_model_path}/cifar10_models/{config.model_name}_{iteration_index + 1 }.pth')
        print("Model Saved Sucessfully")

    return metrics_history


def fine_tuning_using_illusions(save_dir, trainloader, testloader,config,iteration_index,metrics_history):

    net = Net(num_classes=2).to(config.device)

    if iteration_index == 0:
        checkpoint_path = f"{config.load_model_path}/recon_models/pc_recon_t10_uniform_15.pth"
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
        net.load_state_dict(torch.load(f'{config.load_model_path}/illusion_models/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))

    metrics_history=illusion_pc_training(net,trainloader,testloader,"fine_tuning",config,metrics_history)

    if train_bool == True:
        torch.save(net.state_dict(), f'{config.save_model_path}/illusion_models/{config.model_name}_{iteration_index + 1 }.pth')
        print("Model Saved Sucessfully")


    return metrics_history



def decide_training_model(condition,save_dir, trainloader, testloader, config,iteration_index,metrics_history):
    cond_to_func={
            "recon_pc_train":lambda: training_using_reconstruction_and_predicitve_coding(save_dir, trainloader, testloader, config,iteration_index,metrics_history),
            "fine_tuning_classification": lambda:fine_tuning_using_classification(save_dir, trainloader, testloader,config,iteration_index,metrics_history),
            "random_network_testing": lambda:reconstruction_testing_on_random_network(save_dir, trainloader, testloader,config,iteration_index,metrics_history),
            "illusion_train": lambda:fine_tuning_using_illusions(save_dir, trainloader, testloader,config,iteration_index,metrics_history),
            "illusion_train": lambda:fine_tuning_using_illusions(save_dir, trainloader, testloader,config,iteration_index,metrics_history),
            "recon_comparison": lambda:recon_vs_original(testloader, config,8,iteration_index,metrics_history)}

    result=cond_to_func[condition]()

    return result

def cifar_testing(trainloader,testloader,config,iteration_index):
    net = Net(num_classes=10).to(config.device)
    net.load_state_dict(torch.load(f'{config.load_model_path}/cifar10_models/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))
    results=class_pc_training(net,trainloader,testloader,"test",config,iteration_index)
    return results

def cifar_testing_illusion_trained_model(trainloader,testloader,config,iteration_index):
    net = Net(num_classes=2).to(config.device)
    net.load_state_dict(torch.load(f'{config.load_model_path}/illusion_models/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))
    results=illusion_pc_training_custom(net,trainloader,testloader,"test",config,iteration_index)
    return results

def illusion_testing(trainloader,testloader,config,iteration_index):
    net = Net(num_classes=2).to(config.device)
   # checkpoint_path = f"{config.load_model_path}/{config.model_name}.pth"
   # checkpoint = torch.load(checkpoint_path, map_location=config.device,weights_only=True)
   # net.conv1.load_state_dict(checkpoint["conv1"])
   # net.conv2.load_state_dict(checkpoint["conv2"])
   # net.conv3.load_state_dict(checkpoint["conv3"])
   # net.conv4.load_state_dict(checkpoint["conv4"])
   # net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
   # net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
   # net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
   # net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])

    net.load_state_dict(torch.load(f'{config.load_model_path}/illusion_models/{config.model_name}_{iteration_index}.pth',map_location=config.device,weights_only=True))
    results=illusion_pc_training(net, trainloader, testloader,"test", config)

    return results


def get_metrics_initialize(train_cond):
	
    if train_cond == "recon_pc_train":
       metrics_history = {'train_loss': [], 'test_loss': []}
    if train_cond == "fine_tuning_classification":
       metrics_history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],'train_recon_loss':[],'recon_test_loss':[]}
    if train_cond == "illusion_train":
       metrics_history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],'train_recon_loss':[],'recon_test_loss':[]}
    
    return metrics_history

def main(config):
    save_dir = os.path.join("result_folder", f"Seed_{config.seed}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"Accuracy_Stats_{config.seed}.txt")

    trainloader, testloader,validationloader= train_test_loader(config.datasetpath,config.illusion_dataset_bool,config)

    accuracy_transfer=True
    
    metrics_history=get_metrics_initialize(config.training_condition)
    
    for iteration_index in range(config.iterations):
        if config.training_condition == None:
            break
        print(f"The Iteration{iteration_index}:")
        print("================================")
        metrics_history=decide_training_model(config.training_condition, save_dir, trainloader, testloader, config,iteration_index,metrics_history)

     #✅ ADD THIS: Save metrics and plot after all epochs
    from eval_and_plotting import save_training_metrics, plot_training_curves

    print("\n" + "="*60)
    print_status = lambda msg, status: print(f"{'✓' if status=='success' else 'ℹ'} {msg}")
    print_status("Saving training metrics...", "info")

    save_training_metrics(metrics_history,TRAINING_DIR, config.model_name)
    plot_training_curves(metrics_history, TRAINING_DIR, config.model_name)

    print_status("Training complete! Metrics and plots saved.", "success")
    print("="*60 + "\n")

    if config.illusion_dataset_bool == True:
        accuracy_transfer=illusion_testing(trainloader,validationloader,config,2)
    else:
        #accuracy_transfer=cifar_testing(trainloader,testloader,config,15)
        accuracy_transfer=cifar_testing_illusion_trained_model(trainloader,testloader,config,15)

    return accuracy_transfer
        
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

    main(config)













