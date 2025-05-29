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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.fc3_fb = nn.Linear(10, 84)
        self.fc2_fb = nn.Linear(84, 120)
        self.fc1_fb = nn.Linear(120, 16*5*5)
        self.deconv2_fb = nn.ConvTranspose2d(16, 6, 5, stride=1, padding=0)
        self.unpool1 = nn.Upsample(
            size=(14, 14), mode='bilinear', align_corners=False)
        self.deconv1_fb = nn.ConvTranspose2d(
            6, 3, 5, stride=2, padding=2, output_padding=1)
        self.final_upsample = nn.Upsample(
            size=(32, 32), mode='bilinear', align_corners=False)

    def feedforward_pass(self, x):

        ft_AB = self.pool(F.relu(self.conv1(x)))
        ft_BC = self.pool(F.relu(self.conv2(ft_AB)))
        ft_BC = torch.flatten(ft_BC, 1)  # Flatten all dimensions except batch
        ft_CD = F.relu(self.fc1(ft_BC))
        ft_DE = F.relu(self.fc2(ft_CD))
        output = self.fc3(ft_DE)

        return ft_AB, ft_BC, ft_CD, ft_DE, output
