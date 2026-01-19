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
    - Training: 70% of basic shapes + random (stratified by should_see for random)
    - Validation: 30% of basic shapes + random (stratified by should_see for random)
    - Testing: Validation set + matched samples from all_in/all_out
    """
    
    if illusion_bool == "custom_illusion_dataset":
        from customdataset import SquareDataset
        import torchvision.transforms as transforms
        if config.recon_datasetpath == "stl10":
            transform = transforms.Compose([
            transforms.Resize((96, 96)),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        elif config.recon_datasetpath == "cifar10":
            transform = transforms.Compose([
            transforms.Resize((32, 32)),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        else: 
            transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

        DATA_DIR = "data/visual_illusion_dataset"
        
        # ----------------------------------------------------------------
        # Load ALL 8 classes together
        # ----------------------------------------------------------------
        ALL_CLASSES = ["square", "rectangle", "trapezium", "triangle", "hexagon", 
                       "random", "all_in", "all_out"]
        BASIC_CLASSES = ["square", "rectangle", "trapezium", "triangle", "hexagon", "random"]
        
        full_dataset = SquareDataset(
            os.path.join(DATA_DIR, "dataset_metadata.csv"), 
            DATA_DIR,
            classes_for_use=ALL_CLASSES,
            transform=transform
        )
        
        print(f"Full dataset: {len(full_dataset)} samples")
        print(f"Class mapping (ALL classes): {full_dataset.class_to_idx}")
        
        # ----------------------------------------------------------------
        # STEP 1: Separate indices by class
        # ----------------------------------------------------------------
        basic_shape_indices = {cls: [] for cls in ["square", "rectangle", "trapezium", 
                                                    "triangle", "hexagon"]}
        random_indices_by_should_see = {}
        illusion_indices = {"all_in": [], "all_out": []}
        
        for i in range(len(full_dataset)):
            _, _, cls_name, should_see = full_dataset[i]
            
            if cls_name in basic_shape_indices:
                basic_shape_indices[cls_name].append(i)
            elif cls_name == "random":
                if should_see not in random_indices_by_should_see:
                    random_indices_by_should_see[should_see] = []
                random_indices_by_should_see[should_see].append(i)
            elif cls_name in illusion_indices:
                illusion_indices[cls_name].append(i)


        # ----------------------------------------------------------------
        # OPTIONAL: Downsample random class to match basic shape count
        # ----------------------------------------------------------------
        TARGET_RANDOM_TOTAL = len(basic_shape_indices["square"])  # 4608
        NUM_SHOULD_SEE = len(random_indices_by_should_see)

        samples_per_should_see = TARGET_RANDOM_TOTAL // NUM_SHOULD_SEE  # 921

        rng = np.random.default_rng(config.seed)

        downsampled_random_indices_by_should_see = {}

        for should_see_cls, indices in random_indices_by_should_see.items():
            indices = np.array(indices)
            sampled = rng.choice(
                  indices,
                  size=samples_per_should_see,
                  replace=False
                  )
            downsampled_random_indices_by_should_see[should_see_cls] = sampled.tolist()

        random_indices_by_should_see = downsampled_random_indices_by_should_see

        print(f"\nInitial class distribution:")
        for cls in basic_shape_indices:
            print(f"  {cls:15s}: {len(basic_shape_indices[cls])} samples")
        print(f"  {'random':15s}: {sum(len(v) for v in random_indices_by_should_see.values())} samples")
        for see_cls, indices in random_indices_by_should_see.items():
            print(f"    → should_see '{see_cls}': {len(indices)}")
        for cls in illusion_indices:
            print(f"  {cls:15s}: {len(illusion_indices[cls])} samples")
        
        
        # ----------------------------------------------------------------
        # STEP 2: Split basic shapes 70-30
        # ----------------------------------------------------------------
        train_indices = []
        val_indices = []
        
        for cls_name, indices in basic_shape_indices.items():
            indices = np.array(indices)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.3,
                random_state=config.seed
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
        
        # ----------------------------------------------------------------
        # STEP 3: Split random 70-30 STRATIFIED by should_see
        # ----------------------------------------------------------------
        for should_see_cls, indices in random_indices_by_should_see.items():
            indices = np.array(indices)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.3,
                random_state=config.seed
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        
        print(f"\nTraining set: {len(train_indices)} samples")
        print(f"Validation set: {len(val_indices)} samples")
        
        # Analyze validation set to determine illusion sampling
        val_class_counts = {}
        val_should_see_counts = {}
        for idx in val_indices:
            _, _, cls_name, should_see = full_dataset[idx]
            val_class_counts[cls_name] = val_class_counts.get(cls_name, 0) + 1
            if cls_name == "random":
                val_should_see_counts[should_see] = val_should_see_counts.get(should_see, 0) + 1
        
        print(f"Validation class distribution:")
        for cls_name in sorted(val_class_counts.keys()):
            print(f"  {cls_name}: {val_class_counts[cls_name]} samples")
        
        # ----------------------------------------------------------------
        # STEP 4: Sample all_in and all_out for test set
        # ----------------------------------------------------------------
        # Match the should_see distribution from validation's random class
        rng = np.random.default_rng(config.seed)
        test_illusion_indices = []
        
        print(f"\nSampling illusions to match validation random distribution:")
        for illusion_cls in ["all_in", "all_out"]:
            # Group indices by should_see
            indices_by_should_see = {}
            for idx in illusion_indices[illusion_cls]:
                _, _, cls_name, should_see = full_dataset[idx]
                if should_see not in indices_by_should_see:
                    indices_by_should_see[should_see] = []
                indices_by_should_see[should_see].append(idx)
            
            # Sample from each should_see group
            for should_see_cls, count_needed in val_should_see_counts.items():
                available = indices_by_should_see.get(should_see_cls, [])
                if len(available) >= count_needed:
                    sampled = rng.choice(available, count_needed, replace=False)
                else:
                    print(f"  WARNING: {illusion_cls} has only {len(available)} samples "
                          f"for should_see={should_see_cls}, need {count_needed}")
                    sampled = available
                test_illusion_indices.extend(sampled)
            
            print(f"  {illusion_cls}: sampled {len(test_illusion_indices) // 2} samples")
        
        # ----------------------------------------------------------------
        # STEP 5: Create test set = validation + sampled illusions
        # ----------------------------------------------------------------
        test_indices = np.concatenate([val_indices, test_illusion_indices])
        
        # Verify test set distribution
        test_class_counts = {}
        for idx in test_indices:
            _, _, cls_name, _ = full_dataset[idx]
            test_class_counts[cls_name] = test_class_counts.get(cls_name, 0) + 1
        
        print(f"\nTest set: {len(test_indices)} samples")
        print(f"Test set class distribution:")
        for cls_name in sorted(test_class_counts.keys()):
            print(f"  {cls_name}: {test_class_counts[cls_name]} samples")
        
        # ----------------------------------------------------------------
        # Create DataLoaders
        # ----------------------------------------------------------------
        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)
        
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

    elif illusion_bool == "cifar10":
        # CIFAR-10 dataset (reconstruction training)
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root="/home/ajinkyar/datasets",
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
            root="/home/ajinkyar/datasets",
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

    elif illusion_bool == "stl10":
        # STL-10 unlabeled dataset (96x96 -> 128x128 for reconstruction)
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import random_split
        
        # Transform: Resize to 128x128 + normalize
        # Using STL-10 normalization values
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                               (0.2603, 0.2566, 0.2713))
        ])
        
        # Load unlabeled split (100,000 images)
        unlabeled_dataset = torchvision.datasets.STL10(
            root="/home/ajinkyar/datasets",
            split='unlabeled',
            download=True,
            transform=transform
        )
        
        print(f"\nLoaded STL-10 unlabeled dataset: {len(unlabeled_dataset)} images")
        
        # Split into train (80%) and test (20%)
        train_size = int(0.8 * len(unlabeled_dataset))
        test_size = len(unlabeled_dataset) - train_size
        
        trainset, testset = random_split(
            unlabeled_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
        
        print(f"Train split: {len(trainset)} images")
        print(f"Test split: {len(testset)} images")
        
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        validationloader = None
        
        return trainloader, testloader, validationloader



def recon_training_cifar(trainloader, testloader,config,metrics_history,model_name):
    net = Net(num_classes=config.classification_neurons).to(config.device)

    metrics_history=recon_pc_training(net,trainloader,testloader,"train",config,metrics_history,model_name)

    print("Model Save Sucessfully")

    return metrics_history


def classification_training_shapes(class_trainloader,class_validationloader,class_testingloader,recon_trainingloader,config,metrics_history,model_name):

    net = Net(num_classes=config.classification_neurons,input_size=128).to(config.device)
    
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


def illusion_testing(class_trainloader, class_validationloader, class_testingloader,
                     recon_trainingloader, config, metrics_history, model_name):
    
    net = Net(num_classes=config.classification_neurons,input_size=128).to(config.device)
    
    # Extract base model info from config
    base_model_name = config.model_name  # e.g., "pc_recon10_Uniform_seed42_chk15_class_t10_Uniform_seed42"
    checkpoint_epoch = config.checkpoint_epoch  # Which epoch checkpoint to use
    
    checkpoint_path = f"{config.load_model_path}/classification_models/{base_model_name}_epoch{checkpoint_epoch}.pth"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # ❌ WRONG: Your code tries to load the entire dict directly
    # net.load_state_dict(torch.load(checkpoint_path, map_location=config.device, weights_only=False))
    
    # ✅ CORRECT: Load checkpoint dict first, then extract state_dicts
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    
    # Load each layer's state dict individually
    net.conv1.load_state_dict(checkpoint["conv1"])
    net.conv2.load_state_dict(checkpoint["conv2"])
    net.conv3.load_state_dict(checkpoint["conv3"])
    net.conv4.load_state_dict(checkpoint["conv4"])
    net.fc1.load_state_dict(checkpoint["fc1"])
    net.fc2.load_state_dict(checkpoint["fc2"])
    net.fc3.load_state_dict(checkpoint["fc3"])
    net.deconv1_fb.load_state_dict(checkpoint["deconv1_fb"])
    net.deconv2_fb.load_state_dict(checkpoint["deconv2_fb"])
    net.deconv3_fb.load_state_dict(checkpoint["deconv3_fb"])
    net.deconv4_fb.load_state_dict(checkpoint["deconv4_fb"])
    
    print(f"Checkpoint loaded successfully from epoch {checkpoint_epoch}")
    
    # Run testing
    metrics_history = illusion_pc_training(
        net, 
        class_trainloader, 
        class_validationloader, 
        class_testingloader, 
        recon_trainingloader, 
        "test", 
        config, 
        metrics_history, 
        model_name
    )
    
    return metrics_history



def get_metrics_initialize(train_cond):
	
    if train_cond == "recon_pc_train":
       metrics_history = {'train_loss': [], 'test_loss': []}
    elif train_cond == "illusion_testing":
       # ✅ FIX: Test mode doesn't produce training metrics
       metrics_history = {}
    else :
       metrics_history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],'illusory_datset_recon_loss':[],'cifar10_dataset_recon_loss':[]}

    return metrics_history


def decide_training_model(config,metrics_history,model_name):
    recon_training_lr,recon_validation_lr,_=train_test_loader(config.recon_datasetpath,config)
    class_training_lr,class_validation_lr,class_testing_lr=train_test_loader(config.classification_datasetpath,config)

    cond_to_func={
            "recon_pc_train":lambda: recon_training_cifar(recon_training_lr,recon_validation_lr,config,metrics_history,model_name),
            "classification_training_shapes": lambda:classification_training_shapes(class_training_lr,class_validation_lr,class_testing_lr,recon_training_lr,config,metrics_history,model_name),
	    "illusion_testing": lambda:illusion_testing(class_training_lr,class_validation_lr,class_testing_lr,recon_training_lr,config,metrics_history,model_name),
    }

    result=cond_to_func[config.training_condition]()

    return result


def main(config, model_name=None):
    from model_tracking import get_tracker
    from utils import find_seed_siblings
    
    # Update status to training
    if model_name:
        tracker = get_tracker()
        tracker.update_status(model_name, "training")
    
    set_seed(config.seed)
    
    metrics_history = get_metrics_initialize(config.training_condition)
    metrics_history = decide_training_model(config, metrics_history, model_name)
    
    # Save individual model metrics
    from eval_and_plotting import plot_training_metrics
    if config.training_condition != "illusion_testing":
        plot_training_metrics(metrics_history, model_name, config)
    
    # Update status to completed
    if model_name:
        tracker.update_status(model_name, "completed")
        tracker.update_metrics(model_name, metrics_history)
        




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





























