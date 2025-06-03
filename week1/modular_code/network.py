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
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsample_nearest=nn.Upsample(scale_factor=2,mode='nearest')

    def feedforward_pass(self, x):

        ft_AB = self.conv1(x)
        pooled_ft_AB,indices_AB=self.pool(F.relu(ft_AB))
        ft_BC = self.conv2(pooled_ft_AB)
        pooled_ft_BC,indices_BC=self.pool(F.relu(ft_BC))
        ft_BC_flat = torch.flatten(pooled_ft_BC, 1)  # Flatten all dimensions except batch
        ft_CD = self.fc1(ft_BC_flat)
        relu_CD=F.relu(ft_CD)
        ft_DE = self.fc2(relu_CD)
        relu_DE=F.relu(ft_DE)
        output = self.fc3(relu_DE)

        return ft_AB, ft_BC, ft_CD, ft_DE, output,indices_AB,indices_BC
    
    def feedback_pass(self,output,indices_AB,indices_BC,ft_AB,ft_BC,ft_CD,ft_DE):   

        ft_ED=self.fc3_fb(output)
        ft_DC=self.fc2_fb(ft_DE)
        ft_CB=self.fc1_fb(ft_CD)
        ft_CB = ft_CB.view(-1, 16, 8, 8)
        ft_CB=self.upsample_nearest(ft_CB)
        # This matches the forward pass structure correctly
        ft_BA = self.deconv2_fb(self.upsample(ft_BC))        
        x = self.deconv1_fb(ft_AB)              # Final deconv

        return ft_BA, ft_CB, ft_DC, ft_ED, x
