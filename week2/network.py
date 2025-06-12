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
        self.conv3= nn.Conv2d(in_channels=16,out_channels= 64,kernel_size= 5,stride=1,padding=2)
        self.fc1 = nn.Linear(64*4*4, 84)
        self.fc2 = nn.Linear(84, 10)

        self.fc2_fb = nn.Linear(10, 84)
        self.fc1_fb = nn.Linear(84, 64*4*4)
        self.deconv3_fb=nn.ConvTranspose2d(in_channels=64,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.deconv2_fb=nn.ConvTranspose2d(in_channels=16,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.deconv1_fb=nn.ConvTranspose2d(in_channels=6,out_channels=3,kernel_size=5,stride=1,padding=2)
        self.unpool=nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsample_nearest=nn.Upsample(scale_factor=2,mode='nearest')

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

    def predictive_coding_pass(self,x,ft_AB,ft_BC,ft_CD,ft_DE,beta,gamma,alpha,batch_size):

        gamma_AB_fwd,gamma_BC_fwd,gamma_CD_fwd=gamma

        beta_AB_bck,beta_BC_bck,beta_CD_bck=beta

        alpha_AB,alpha_BC,alpha_CD=alpha

        errorB=nn.functional.mse_loss(self.deconv1_fb(ft_AB),x)
        
        reconstructionB=torch.autograd.grad(errorB,ft_AB,retain_graph=True)[0]
        
        #This scaling is done by the factor sqrt((k^2/C)) ref: https://proceedings.neurips.cc/paper_files/paper/2021/file/75c58d36157505a600e0695ed0b3a22d-Supplemental.pdf supplement A. The reason why we do this scaling is because by simply dividing it by the number of neurons wouldn't be helpful since not all of them are involved in the receptive field so we should also take into consideration the elements in the receptive and then take their ratio with respect to the the total number of neurons.
        

        scalingB=np.round(np.sqrt(np.square(np.prod(x.shape[1:])) / np.prod(self.deconv1_fb(torch.rand_like(ft_AB)).shape[1:])))
        
        #The predictive coding has three main terms the forward the backward and the error. Forward is controlled by gamma and backward is controlled by beta and the error gradient is controlled by alpha and the memory term is controlled by 1 - gamma - beta
        
        ft_AB_pc = gamma_AB_fwd*self.conv1(x) + (1-gamma_AB_fwd-beta_AB_bck) * ft_AB + beta_AB_bck*self.deconv2_fb(self.upsample(ft_BC))-alpha_AB*scalingB*batch_size*reconstructionB

        errorC=nn.functional.mse_loss(self.deconv2_fb(self.upsample(ft_BC)),ft_AB)

        reconstructionC=torch.autograd.grad(errorC,ft_BC,retain_graph=True)[0]

        pooled_ft_AB_pc,indices_AB=self.pool(F.relu(ft_AB_pc))

        #print("Shape of AB",ft_AB.shape[1:])
        #print("Shape Of Convolutional Layer",self.deconv2_fb(torch.rand_like(ft_BC)).shape[1:])

        scalingC=np.round(np.sqrt(np.square( np.prod(ft_AB.shape[1:]))/np.prod(self.deconv2_fb(torch.rand_like(ft_BC)).shape[1:])))

        ft_BC_pc = gamma_BC_fwd*self.conv2(pooled_ft_AB_pc) + (1-gamma_BC_fwd-beta_BC_bck) * ft_BC + beta_BC_bck*self.deconv3_fb(self.upsample(ft_CD))+alpha_BC*scalingC*batch_size*reconstructionC
        
        pooled_ft_BC_pc,indices_BC=self.pool(F.relu(ft_BC_pc))
        
        errorD=nn.functional.mse_loss(self.deconv3_fb(self.upsample(ft_CD)),ft_BC)

        reconstructionD=torch.autograd.grad(errorD,ft_CD,retain_graph=True)[0]

        scalingD= np.round(np.sqrt(np.square(np.prod(ft_BC.shape[1:])) / np.prod(self.deconv3_fb(torch.rand_like(ft_CD)).shape[1:])))
        ft_CD_pc=gamma_CD_fwd * self.conv3(pooled_ft_BC_pc) + (1-gamma_CD_fwd) * ft_CD + alpha_CD*scalingD*batch_size*reconstructionD

        pooled_ft_CD,_=self.pool(F.relu(ft_CD_pc))

        ft_CD_flat=torch.flatten(pooled_ft_CD,1)
        
        ft_DE = self.fc1(ft_CD_flat)
        
        relu_DE=F.relu(ft_DE)
        
        output=self.fc2(relu_DE)

        return output,ft_AB_pc,ft_BC_pc,ft_CD_pc,ft_DE
