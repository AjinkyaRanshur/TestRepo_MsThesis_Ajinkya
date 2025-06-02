import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels= 6,kernel_size= 5,stride=1,padding=2)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels= 16,kernel_size= 5,stride=1,padding=2)
        self.fc1 = nn.Linear(16*8*8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.fc3_fb = nn.Linear(10, 84)
        self.fc2_fb = nn.Linear(84, 120)
        self.fc1_fb = nn.Linear(120, 16*8*8)
        self.deconv2_fb=nn.ConvTranspose2d(in_channels=16,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.deconv1_fb=nn.ConvTranspose2d(in_channels=6,out_channels=3,kernel_size=5,stride=1,padding=2)
        self.unpool=nn.MaxUnpool2d(kernel_size=2, stride=2)


    def feedforward_pass(self, x):

        ft_AB,indices_AB = self.pool(F.relu(self.conv1(x)))
        ft_BC,indices_BC = self.pool(F.relu(self.conv2(ft_AB)))
        ft_BC_flat = torch.flatten(ft_BC, 1)  # Flatten all dimensions except batch
        ft_CD = F.relu(self.fc1(ft_BC_flat))
        ft_DE = F.relu(self.fc2(ft_CD))
        output = self.fc3(ft_DE)

        return ft_AB, ft_BC, ft_CD, ft_DE, output,indices_AB,indices_BC
    
    def feedback_pass(self,output,indices_AB,indices_BC,ft_AB,ft_BC,ft_CD,ft_DE):   

        ft_ED=self.fc3_fb(output)
        ft_DC=F.relu(self.fc2_fb(ft_DE))
        ft_CB=F.relu(self.fc1_fb(ft_CD))
        ft_CB = ft_CB.view(-1, 16, 8, 8)
        # This matches the forward pass structure correctly
        ft_BC_unpool = F.relu(self.unpool(ft_BC, indices_BC))  # Reconstruct from 8x8 to 16x16
        ft_BA = self.deconv2_fb(ft_BC_unpool)          # Then apply deconv
        ft_BA_unpooled = F.relu(self.unpool(ft_BA, indices_AB)) # Reconstruct from 16x16 to 32x32
        x = self.deconv1_fb(ft_BA_unpooled)              # Final deconv

        return ft_BA, ft_CB, ft_DC, ft_ED, x
