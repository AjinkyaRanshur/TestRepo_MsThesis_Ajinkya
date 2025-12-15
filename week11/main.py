# Standard Machine learning Libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset,random_split,DataLoader,Subset
from torch.utils.tensorboard import SummaryWriter

# Functions Imports for ease of life :))
from network import Net as Net
#from pc_train import class_pc_training
from recon_pc_train import recon_pc_training
from illusion_pc_train import illusion_pc_training
from customdataset import SquareDataset
#from eval_and_plotting import save_training_metrics, plot_training_curve

# Utility Functions
import random
import os
import sys
import time
import argparse
import importlib
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split



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


def train_test_loader(illusion_bool,config):
    # Normalizing the images
    #Andrea's nromalization is different figure out why    
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
    if illusion_bool == "illusion":
        DATA_DIR = config.classification_datasetpath
        full_basic_random=SquareDataset(os.path.join(DATA_DIR, "dataset_metadata.csv"), DATA_DIR,classes_for_use=["square","rectangle","trapezium","triangle","hexagon","random"],transform=transform)
        
        labels = np.array([full_basic_random[i][1] for i in range(len(full_basic_random))])

        train_idx, val_idx = train_test_split(
         np.arange(len(full_basic_random)),
         test_size=0.2,
         random_state=42,
         stratify=labels
         )

        train_set = Subset(full_basic_random, train_idx)
        val_set   = Subset(full_basic_random, val_idx)

        ALL_IN_OUT = ["all_in", "all_out"]

        all_in_out_dataset = SquareDataset(
        os.path.join(DATA_DIR, "dataset_metadata.csv"),
        DATA_DIR,
        classes_for_use=ALL_IN_OUT,
        transform=transform
        )

        # Count how many per class are needed:
        # number needed from each = size of val_set / 6
        # because val_set contains 6 classes (5 basic + random)
        num_per_class = len(val_set) // 6

        # Filter dataset indices per class
        all_in_indices  = [i for i in range(len(all_in_out_dataset))
                   if all_in_out_dataset[i][2] == "all_in"]

        all_out_indices = [i for i in range(len(all_in_out_dataset))
                   if all_in_out_dataset[i][2] == "all_out"]

        # ✅ Add debug prints to verify
        print(f"Found {len(all_in_indices)} all_in images")
        print(f"Found {len(all_out_indices)} all_out images")
        print(f"Need {num_per_class} images per class")

        # Randomly sample so test classes balanced
        rng = np.random.default_rng(config.seed)
        chosen_all_in  = rng.choice(all_in_indices,  num_per_class, replace=False)
        chosen_all_out = rng.choice(all_out_indices, num_per_class, replace=False)

        # Build test set by merging:
        # - the 20% basic+random (val_idx)
        # - num_per_class of all_in
        # - num_per_class of all_out
        test_indices = list(val_idx) + chosen_all_in.tolist() + chosen_all_out.tolist()

        # Create unified test dataset by merging two sources
        # Trick: we wrap both datasets inside a CombinedDataset
        class CombinedDataset(Dataset):
              def __init__(self, basic_random, illusion, val_idx, allin_idx, allout_idx):
                  self.basic_random = basic_random
                  self.illusion = illusion
                  self.indices = list(val_idx) + list(allin_idx) + list(allout_idx)
              def __len__(self):
                  return len(self.indices)
              def __getitem__(self, idx):
                  real_idx = self.indices[idx]
                  if real_idx < len(self.basic_random):
                     return self.basic_random[real_idx]
                  else:
                     adj = real_idx - len(self.basic_random)
                     return self.illusion[adj]

        test_set = CombinedDataset(full_basic_random, all_in_out_dataset,
                           val_idx, chosen_all_in, chosen_all_out)

        # ------------------------------------------------------------------
        # 4. Dataloaders
        # ------------------------------------------------------------------
        trainloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        validationloader   = DataLoader(val_set,   batch_size=config.batch_size, shuffle=False)
        testloader  = DataLoader(test_set,  batch_size=config.batch_size, shuffle=False)

    else:
        trainset = torchvision.datasets.CIFAR10(
        root=config.recon_datasetpath,
        train=True,
        download=True,
        transform=transform)

        trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(
        root=config.recon_datasetpath,
        train=False,
        download=True,
        transform=transform)

        testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
        validationloader=0
    return trainloader, testloader,validationloader

def recon_training_cifar(trainloader, testloader,config,metrics_history):
    net = Net(num_classes=config.classification_neurons).to(config.device)

    metrics_history=recon_pc_training(net,trainloader,testloader,"train",config,metrics_history)

    print("Model Save Sucessfully")

    return metrics_history


def classification_training_shapes(class_trainloader,class_validationloader,class_testingloader,recon_trainingloader,config,metrics_history):

    net = Net(num_classes=config.classification_neurons).to(config.device)
    
    # Set to whichever value for using the recon model
    iteration_index=15

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

    metrics_history=illusion_pc_training(class_trainloader,class_validationloader,class_testingloader,recon_trainingloader,"fine_tuning",config,metrics_history)


    return metrics_history



def get_metrics_initialize(train_cond):
	
    if train_cond == "recon_pc_train":
       metrics_history = {'train_loss': [], 'test_loss': []}
    else :
       metrics_history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],'train_recon_loss':[],'test_recon_loss':[]}

    return metrics_history


def decide_training_model(config,metrics_history):
    recon_training_lr,recon_validation_lr,_=train_test_loader("reconstruction",config)
    class_training_lr,class_validation_lr,class_testing_lr=train_test_loader("illusion",config)

    cond_to_func={
            "recon_pc_train":lambda: recon_training_cifar(recon_training_lr,recon_validation_lr,config,metrics_history),
            "classification_training_shapes": lambda:classification_training_shapes(class_training_lr,class_validation_lr,class_testing_lr,recon_training_lr,config,metrics_history),
    }

    result=cond_to_func[config.training_condition]()

    return result


def main(config,model_id=None):
    
    from model_tracking import get_tracker
    
    # Update status to training
    if model_id:
        tracker = get_tracker()
        tracker.update_status(model_id, "training")
    
    metrics_history=get_metrics_initialize(config.training_condition)
    metrics_history= decide_training_model(config,metrics_history)
    set_seed(config.seed)
   
    # Update status to completed and save metrics
    if model_id:
        tracker.update_status(model_id, "completed")
    


def load_config(config_name):
    return importlib.import_module(config_name)

# This line ensures safe multiprocessing
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None, 
                       help="Model ID from tracking system")
    args = parser.parse_args()

    # Add config directory to Python path
    sys.path.append(os.path.abspath("configs"))

    config = load_config(args.config)

    main(config,args.model_name)






























