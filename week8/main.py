# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from network import Net as Net
from fwd_train import feedfwd_training
from back_train import feedback_training
from pc_train import class_pc_training
from recon_pc_train import recon_pc_training
from eval_and_plotting import eval_pc_accuracy,recon_pc_loss
from illusion_pc_train import illusion_pc_training
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


class SquareDataset(Dataset):
    def __init__(self, csv_file, img_dir, classes_for_use, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata[self.metadata['Class'].isin(
            classes_for_use)]
        self.img_dir = img_dir
        self.transform = transform
        self.metadata['Label'] = self.metadata['Class'].apply(
            lambda x: 1 if x == "Square" else 0)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['Class'],
                                self.metadata.iloc[idx]['filename'])
        image = Image.open(img_name).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(
            self.metadata.iloc[idx]['Label'],
            dtype=torch.float32)
        cls_name = self.metadata.iloc[idx]['Class']
        return image, label, cls_name


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

class FilteredShapeDataset(Dataset):
    def __init__(self, txt_path, transform=None, filter_classes=None):
        """
        Dataset that can filter specific classes

        Args:
            txt_path: path to your data file
            transform: image transforms
            filter_classes: list of classes to include (e.g., [0, 1] for training)
                          None means include all classes (for testing)
        """
        self.data = []
        self.transform = transform

        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    img_path = parts[0]
                    label = int(parts[1])

                    # Filter classes if specified
                    if filter_classes is None or label in filter_classes:
                        shape_type = int(parts[2]) if parts[2] != '-1' else -1
                        x1, y1, x2, y2 = map(float, parts[3:7])
                        noise1, noise2 = map(float, parts[7:9])

                        self.data.append({
                            'path': img_path,
                            'label': label,
                            'shape_type': shape_type,
                            'bbox': [x1, y1, x2, y2],
                            'noise': [noise1, noise2]
                        })

        print(f"Loaded {len(self.data)} samples")
        if filter_classes:
            print(f"Filtered to classes: {filter_classes}")
            # Count samples per class
            class_counts = {}
            for item in self.data:
                label = item['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            print("Class distribution:", class_counts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, item['label']

def train_test_loader(datasetpath,illusion_bool):
    # Normalizing the images
    #Andrea's nromalization is different figure out why

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    if illusion_bool == True:
        DATA_DIR = datasetpath
        trainset=SquareDataset(os.path.join(DATA_DIR, "metadata.csv"), DATA_DIR,classes_for_use=["Square", "Random"],transform=transform)
        trainloader=torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        testset=SquareDataset(os.path.join(DATA_DIR, "metadata.csv"), DATA_DIR,classes_for_use=["Square", "Random", "All-in", "All-out"],transform=transform)
        testloader=torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        

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

    for iteration_index in range(20):
        print(f"The Iteration{iteration_index}:")
        print("================================")
        train_bool=recon_pc_training(net,trainloader,testloader,"fine_tuning",config)
        if train_bool == True:
            torch.save(net.state_dict(), f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
            print("Model Saved Sucessfully")

    return None


def fine_tuning_using_classification(net,save_dir, trainloader, testloader,config):
    
    for iteration_index in range(20):
        print(f"The Iteration{iteration_index}:")
        print("================================")
        net.load_state_dict(torch.load(f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',
        map_location=config.device,weights_only=True))
        train_bool=recon_pc_training(net,trainloader,testloader,"fine_tuning",config)
        if train_bool == True:
            torch.save(net.state_dict(), f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
            print("Model Saved Sucessfully")

    return train_bool

def fine_tuning_using_illusions(net,save_dir, trainloader, testloader,config):

    for iteration_index in range(2):
        print(f"The Iteration{iteration_index}:")
        print("================================")
        net.load_state_dict(torch.load(f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',
        map_location=config.device,weights_only=True))
        train_bool=illusioin_pc_training(net,trainloader, testloader,"fine_tuning",config)
        if train_bool == True:
            torch.save(net.state_dict(), f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
            print("Model Saved Sucessfully")

    return train_bool

def recon_vs_original(net, dataloader, config, n_images=8, epoch=None, phase="train", iteration_index=15):
    """
    Compare original images with their reconstructions from the network.
    
    Args:
        net: Neural network model
        dataloader: Data loader for images
        config: Configuration object with device, model paths, etc.
        n_images: Number of images to compare (default 8)
        epoch: Current epoch number (optional)
        phase: Training phase ("train" or "test")
        iteration_index: Model iteration to load
    """
    
    # Load the trained model
    try:
        net.load_state_dict(torch.load(
            f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',
            map_location=config.device,
            weights_only=True))
        print(f"✓ Loaded model from iteration {iteration_index}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {config.load_model_path}/{config.model_name}_{iteration_index}.pth")
        return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

    net.eval()

    # Get a batch of images
    try:
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
    except StopIteration:
        print("✗ No data available in dataloader")
        return False

    # Ensure we don't exceed available images
    actual_batch_size = images.size(0)
    n_images = min(n_images, actual_batch_size)
    
    images = images[:n_images].to(config.device)
    labels = labels[:n_images]

    with torch.no_grad():
        # Initialize feature tensors with correct batch size
        ft_AB = torch.zeros(n_images, 6, 32, 32).to(config.device)
        ft_BC = torch.zeros(n_images, 16, 16, 16).to(config.device)
        ft_CD = torch.zeros(n_images, 32, 8, 8).to(config.device)
        ft_DE = torch.zeros(n_images, 64, 4, 4).to(config.device)

        # Forward pass
        try:
            ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG, output = net.feedforward_pass(
                images, ft_AB, ft_BC, ft_CD, ft_DE
            )
        except Exception as e:
            print(f"✗ Error in feedforward pass: {e}")
            return False

        # Feedback pass for reconstruction
        try:
            ft_BA, ft_CB, ft_DC, ft_ED, ft_FE, ft_GF, reconstructed = net.feedback_pass(
                output, ft_AB, ft_BC, ft_CD, ft_DE, ft_EF, ft_FG
            )
        except Exception as e:
            print(f"✗ Error in feedback pass: {e}")
            return False

        # Normalize images for proper visualization
        # Move to CPU and detach from computation graph
        original_imgs = images.detach().cpu()
        reconstructed_imgs = reconstructed.detach().cpu()
        
        # Denormalize images (reverse CIFAR-10 normalization)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        
        original_imgs = original_imgs * std + mean
        reconstructed_imgs = reconstructed_imgs * std + mean
        
        # Clamp to valid range [0, 1]
        original_imgs = torch.clamp(original_imgs, 0, 1)
        reconstructed_imgs = torch.clamp(reconstructed_imgs, 0, 1)

        # Create comparison grid
        comparison_images = torch.zeros(n_images * 2, 3, 32, 32)

        # Alternate original and reconstructed images
        for i in range(n_images):
            comparison_images[i * 2] = original_imgs[i]           # Original
            comparison_images[i * 2 + 1] = reconstructed_imgs[i]  # Reconstructed
    
    # Create grid: 2 rows x n_images columns
    grid = vutils.make_grid(comparison_images, nrow=n_images, padding=2, normalize=False)
    
    # Convert to numpy for matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(max(15, n_images * 2), 6))
    ax.imshow(grid_np)
    ax.axis('off')
    
    # Add title and labels
    epoch_str = f"Epoch {epoch}" if epoch is not None else f"Iteration {iteration_index}"
    ax.set_title(f'Image Reconstruction Comparison - {phase.capitalize()} {epoch_str}\n'
                 f'Top Row: Original Images, Bottom Row: Reconstructed Images', 
                 fontsize=14, pad=20)
    
    # Add text annotations for clarity
    ax.text(0.02, 0.15, 'Original', transform=ax.transAxes, fontsize=12, 
            color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    ax.text(0.02, 0.85, 'Reconstructed', transform=ax.transAxes, fontsize=12,
            color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    # Log to wandb
    try:
        wandb.log({
            f"Reconstruction_Comparison/{phase}": wandb.Image(fig, 
                caption=f"Original vs Reconstructed - {phase} {epoch_str}")
        })
        print(f"✓ Logged reconstruction comparison to wandb")
    except Exception as e:
        print(f"✗ Error logging to wandb: {e}")
    
    plt.close(fig)  # Close figure to save memory
    
    # Calculate and log reconstruction loss
    with torch.no_grad():
        recon_loss = F.mse_loss(reconstructed_imgs, original_imgs).item()
        try:
            wandb.log({f"Reconstruction_Loss/{phase}": recon_loss})
            print(f"✓ Reconstruction MSE Loss: {recon_loss:.6f}")
        except Exception as e:
            print(f"✗ Error logging reconstruction loss: {e}")

    return True

def testing_model(net,trainloader,testloader,config,iteration_index):

    net.load_state_dict(
    torch.load(
        f'{config.load_model_path}/{config.model_name}_{iteration_index}.pth',
        map_location=config.device,
         weights_only=True))

    class_pc_training(net,trainloader,testloader,"test",config,iteration_index)

    return None


def main():
    init_wandb(config.batch_size,config.epochs,config.lr,config.momentum,config.seed,config.device,config.training_condition,config.timesteps,config.gammaset,config.betaset,config.alphaset,config.datasetpath,config.experiment_name,config.noise_type,config.noise_param,config.model_name)

    save_dir = os.path.join("result_folder", f"Seed_{config.seed}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"Accuracy_Stats_{config.seed}.txt")
    net = Net().to(config.device)
    wandb.watch(net, log="all", log_freq=10)
    trainloader, testloader = train_test_loader(config.datasetpath,config.illusion_dataset_bool)





    if config.training_condition == "fine_tuning_classification":
        train_bool = fine_tuning_using_classification(net,save_dir, trainloader, testloader,config)

    if config.training_condition == "random_network_testing":
            train_bool = reconstruction_testing_on_random_network(net,save_dir, trainloader, testloader,config)
    
    if config.training_condition == "illusion_train":
        train_bool = fine_tuning_using_illusions(net,save_dir, trainloader, testloader,config)

    if config.training_condition == "recon_comparison":
        success = recon_vs_original(net, testloader, config, n_images=8, iteration_index=15)
        if not success:
            print("Failed to generate reconstruction comparison")


    for iteration_index in range(20):
        if config.training_condition == None:
            break
        print(f"The Iteration{iteration_index}:")
        print("================================")
        if iteration_index != 0:
            state_dict = torch.load(
            f"{config.load_model_path}/{config.model_name}_{iteration_index}.pth",
            map_location=config.device,weights_only=True
            )
            net.load_state_dict(state_dict, strict=False)


        if config.training_condition == "recon_pc_train":
            train_bool = training_using_reconstruction_and_predicitve_coding(net,save_dir, trainloader, testloader,config)
            if train_bool == True:
                # Specify exactly which layers you want to save
                layers_to_save = [
                'conv1.weight', 'conv1.bias',
                'conv2.weight', 'conv2.bias', 
                'conv3.weight', 'conv3.bias',
                'conv4.weight', 'conv4.bias',
                'deconv1_fb.weight', 'deconv1_fb.bias',
                'deconv2_fb.weight', 'deconv2_fb.bias',
                'deconv3_fb.weight', 'deconv3_fb.bias',
                'deconv4_fb.weight', 'deconv4_fb.bias',
                ]
                # Collect only the selected layers from state_dict
                filtered_state = {k: v for k, v in net.state_dict().items() if k in layers_to_save}
                torch.save(filtered_state,f'{config.save_model_path}/{config.model_name}_{iteration_index + 1 }.pth')
                print("Training Sucessful")


    #accuracy_dict = testing_model(net,trainloader,testloader,config,20)
                

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

