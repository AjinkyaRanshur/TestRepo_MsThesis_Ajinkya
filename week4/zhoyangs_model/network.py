import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PredictiveCoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder1=nn.Conv2d(in_channels=3,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)
        self.encoder2=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)
        self.encoder3=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)

        self.decoder2=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=1,padding=2)
        self.decoder1=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=1,padding=2)
        self.decoder0=nn.Conv2d(in_channels=128,out_channels= 3,kernel_size= 5,stride=1,padding=2)

        self.fc1 = nn.Linear(128*4*4,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,100)

    def feedforward_pass(self, x):

        ft_AB = self.conv1(x)
        pooled_ft_AB,_=self.pool(F.relu(ft_AB))
        ft_BC = self.conv2(pooled_ft_AB)
        pooled_ft_BC,_=self.pool(F.relu(ft_BC))
        ft_CD = self.conv3(pooled_ft_BC)
        pooled_ft_CD,_=self.pool(F.relu(ft_CD))
        ft_CD_flat=torch.flatten(pooled_ft_CD,1)
        ft_DE = self.fc1(ft_CD_flat)
        relu_DE=F.relu(ft_DE)
        output=self.fc2(relu_DE)

        return ft_AB, ft_BC, ft_CD, ft_DE, output
    
    def feedback_pass(self,output,ft_AB,ft_BC,ft_CD,ft_DE):   
        ft_ED=self.fc2_fb(output)
        ft_DC=self.fc1_fb(ft_DE)
        ft_DC=ft_DC.view(-1,64,4,4)
        ft_DC=self.upsample_nearest(ft_DC)
        ft_CB=self.deconv3_fb(self.upsample(ft_CD))
        ft_BA = self.deconv2_fb(self.upsample(ft_BC))        
        x = self.deconv1_fb(ft_AB)              # Final deconv

        return ft_BA, ft_CB, ft_DC, ft_ED, x




