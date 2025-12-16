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

def train_test_loader(illusion_bool, config):
    """
    Creates train/validation/test dataloaders.
    
    For illusion dataset:
    - Training: 80% of basic shapes (square, rectangle, trapezium, triangle, hexagon, random)
    - Validation: 20% of basic shapes (non-overlapping with training)
    - Testing: All of validation set + equal number of all_in and all_out samples
    """
    
    if illusion_bool == "illusion":
        from customdataset import SquareDataset
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        DATA_DIR = config.classification_datasetpath
        
        # ----------------------------------------------------------------
        # Load basic shapes + random (for training/validation)
        # ----------------------------------------------------------------
        BASIC_CLASSES = ["square", "rectangle", "trapezium", "triangle", "hexagon", "random"]
        
        basic_dataset = SquareDataset(
            os.path.join(DATA_DIR, "dataset_metadata.csv"), 
            DATA_DIR,
            classes_for_use=BASIC_CLASSES,
            transform=transform
        )
        
        # Split into train (80%) and validation (20%)
        # Using stratify ensures EQUAL distribution across all 6 classes
        labels = np.array([basic_dataset[i][1] for i in range(len(basic_dataset))])
        
        train_idx, val_idx = train_test_split(
            np.arange(len(basic_dataset)),
            test_size=0.2,
            random_state=config.seed,
            stratify=labels  # This ensures equal class distribution
        )
        
        train_set = Subset(basic_dataset, train_idx)
        val_set = Subset(basic_dataset, val_idx)
        
        print(f"Training set: {len(train_set)} samples")
        print(f"Validation set: {len(val_set)} samples")
        
        # Verify class balance in validation set
        val_class_counts = {}
        for idx in val_idx:
            label = labels[idx]
            class_name = BASIC_CLASSES[label]
            val_class_counts[class_name] = val_class_counts.get(class_name, 0) + 1
        
        print("Validation set class distribution:")
        for class_name, count in sorted(val_class_counts.items()):
            print(f"  {class_name}: {count} samples")
        
        # ----------------------------------------------------------------
        # Load illusion images (all_in, all_out) for testing
        # ----------------------------------------------------------------
        ILLUSION_CLASSES = ["all_in", "all_out"]
        
        illusion_dataset = SquareDataset(
            os.path.join(DATA_DIR, "dataset_metadata.csv"),
            DATA_DIR,
            classes_for_use=ILLUSION_CLASSES,
            transform=transform
        )
        
        # Get indices for each illusion class
        all_in_indices = [i for i in range(len(illusion_dataset))
                         if illusion_dataset[i][2] == "all_in"]
        all_out_indices = [i for i in range(len(illusion_dataset))
                          if illusion_dataset[i][2] == "all_out"]
        
        print(f"Found {len(all_in_indices)} all_in images")
        print(f"Found {len(all_out_indices)} all_out images")
        
        # Sample equal number from each illusion class (same as validation set size)
        num_per_illusion_class = len(val_set) // len(BASIC_CLASSES)
        
        rng = np.random.default_rng(config.seed)
        chosen_all_in = rng.choice(all_in_indices, num_per_illusion_class, replace=False)
        chosen_all_out = rng.choice(all_out_indices, num_per_illusion_class, replace=False)
        
        print(f"Sampling {num_per_illusion_class} images per illusion class")
        
        # ----------------------------------------------------------------
        # Create combined test dataset
        # ----------------------------------------------------------------
        class CombinedTestDataset(Dataset):
            """Combines validation basic shapes with illusion samples"""
            def __init__(self, basic_dataset, illusion_dataset, val_idx, allin_idx, allout_idx):
                self.basic_dataset = basic_dataset
                self.illusion_dataset = illusion_dataset
                
                # Store which indices belong to which dataset
                self.val_idx = list(val_idx)
                self.allin_idx = list(allin_idx)
                self.allout_idx = list(allout_idx)
                
                self.total_len = len(val_idx) + len(allin_idx) + len(allout_idx)
                
            def __len__(self):
                return self.total_len
            
            def __getitem__(self, idx):
                # First part: validation basic shapes
                if idx < len(self.val_idx):
                    real_idx = self.val_idx[idx]
                    return self.basic_dataset[real_idx]
                
                # Second part: all_in samples
                elif idx < len(self.val_idx) + len(self.allin_idx):
                    offset = idx - len(self.val_idx)
                    real_idx = self.allin_idx[offset]
                    return self.illusion_dataset[real_idx]
                
                # Third part: all_out samples
                else:
                    offset = idx - len(self.val_idx) - len(self.allin_idx)
                    real_idx = self.allout_idx[offset]
                    return self.illusion_dataset[real_idx]
        
        test_set = CombinedTestDataset(
            basic_dataset, 
            illusion_dataset,
            val_idx, 
            chosen_all_in, 
            chosen_all_out
        )
        
        print(f"Test set: {len(test_set)} samples")
        print(f"  - {len(val_idx)} basic shapes")
        print(f"  - {len(chosen_all_in)} all_in")
        print(f"  - {len(chosen_all_out)} all_out")
        
        # ----------------------------------------------------------------
        # Create DataLoaders
        # ----------------------------------------------------------------
        trainloader = DataLoader(
            train_set, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        
        validationloader = DataLoader(
            val_set, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        testloader = DataLoader(
            test_set, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        return trainloader, validationloader, testloader

    else:
        # CIFAR-10 dataset (reconstruction training)
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root=config.recon_datasetpath,
            train=True,
            download=True,
            transform=transform
        )
        
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=config.recon_datasetpath,
            train=False,
            download=True,
            transform=transform
        )
        
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        validationloader = None
        
        return trainloader, testloader, validationloader



def recon_training_cifar(trainloader, testloader,config,metrics_history,model_name):
    net = Net(num_classes=config.classification_neurons).to(config.device)

    metrics_history=recon_pc_training(net,trainloader,testloader,"train",config,metrics_history,model_name)

    print("Model Save Sucessfully")

    return metrics_history


def classification_training_shapes(class_trainloader,class_validationloader,class_testingloader,recon_trainingloader,config,metrics_history,model_name):

    net = Net(num_classes=config.classification_neurons).to(config.device)
    
    # Set to whichever value for using the recon model
    # Extract base model info from config
    base_model_name = config.base_recon_model  # e.g., "pc_recon10_Uniform_seed42"
    checkpoint_epoch = config.checkpoint_epoch  # Which epoch checkpoint to use

    checkpoint_path = f"{config.load_model_path}/recon_models/{base_model_name}_{checkpoint_epoch}.pth"
	
  
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device,weights_only=False)


    net.conv1.load_state_dict(checkpoint["conv1"])
    net.conv2.load_state_dict(checkpoint["conv2"])
    net.conv3.load_state_dict(checkpoint["conv3"])
    net.conv4.load_state_dict(checkpoint["conv4"])
    net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
    net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
    net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
    net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])

    print(f"Checkpoint loaded successfully from epoch {checkpoint_epoch}")

    metrics_history=illusion_pc_training(net,class_trainloader,class_validationloader,class_testingloader,recon_trainingloader,"fine_tuning",config,metrics_history,model_name)


    return metrics_history



def illusion_testing(class_trainloader,class_validationloader,class_testingloader,recon_trainingloader,config,metrics_history,model_name):

    net = Net(num_classes=config.classification_neurons).to(config.device)

    # Set to whichever value for using the recon model
    # Extract base model info from config
    base_model_name = config.model_name  # e.g., "pc_recon10_Uniform_seed42"
    checkpoint_epoch = config.checkpoint_epoch  # Which epoch checkpoint to use

    checkpoint_path = f"{config.load_model_path}/classification_models/{base_model_name}_epoch{checkpoint_epoch}.pth"


    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device,weights_only=False)

    print(f"Checkpoint loaded successfully from epoch {checkpoint_epoch}")

    metrics_history=illusion_pc_training(net,class_trainloader,class_validationloader,class_testingloader,recon_trainingloader,"test",config,metrics_history,model_name)


    return metrics_history



def get_metrics_initialize(train_cond):
	
    if train_cond == "recon_pc_train":
       metrics_history = {'train_loss': [], 'test_loss': []}
    else :
       metrics_history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],'train_recon_loss':[],'test_recon_loss':[]}

    return metrics_history


def decide_training_model(config,metrics_history,model_name):
    recon_training_lr,recon_validation_lr,_=train_test_loader("reconstruction",config)
    class_training_lr,class_validation_lr,class_testing_lr=train_test_loader("illusion",config)

    cond_to_func={
            "recon_pc_train":lambda: recon_training_cifar(recon_training_lr,recon_validation_lr,config,metrics_history,model_name),
            "classification_training_shapes": lambda:classification_training_shapes(class_training_lr,class_validation_lr,class_testing_lr,recon_training_lr,config,metrics_history,model_name),
	    "illusion_testing": lambda:illusion_testing(class_training_lr,class_validation_lr,class_testing_lr,recon_training_lr,config,metrics_history,model_name),
    }

    result=cond_to_func[config.training_condition]()

    return result


def main(config,model_name=None):
    
    from model_tracking import get_tracker
    
    # Update status to training
    if model_name:
        tracker = get_tracker()
        tracker.update_status(model_name, "training")
    
    metrics_history=get_metrics_initialize(config.training_condition)

    metrics_history= decide_training_model(config,metrics_history,model_name)
    
    # Save final training metrics plot
    from eval_and_plotting import plot_training_metrics
    plot_training_metrics(metrics_history, model_name, config)

    set_seed(config.seed)
   
    # Update status to completed and save metrics
    if model_name:
        tracker.update_status(model_name, "completed")
    

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





























