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
        ##This network does not have any sort of pooling according to the paper and repo which is very weird
        self.encoder1=nn.Conv2d(in_channels=3,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)
        self.encoder2=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)
        self.encoder3=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)

        self.decoder3=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=1,padding=2)
        self.decoder2=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=1,padding=2)
        self.decoder1=nn.Conv2d(in_channels=128,out_channels= 3,kernel_size= 5,stride=1,padding=2)

        self.dropout_layer = nn.Dropout2d(self.dropout)
        self.bach_normal   = nn.BatchNorm1d(num_features=2048)
        self.fc1 = nn.Linear(128*4*4,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,100)

    def feedforward_pass(self, x):
        #A->(encoder1)->B->(encoder2)->C->(encoder3)->D->(fc1)->E->(fc2)->F->(fc3)->G->(softmax)->output
        ft_AB = self.encoder1(x)
        relu_ft_AB=F.relu(ft_AB)
        ft_BC = self.encoder2(relu_ft_AB)
        relu_ft_BC=F.relu(ft_BC)
        ft_CD = self.encoder3(relu_ft_BC)
        relu_ft_CD=F.relu(ft_CD)

        return ft_AB,ft_BC,ft_CD,relu_ft_CD
    
    def feedback_pass(self,ft_AB,ft_BC,ft_CD):
        #A->(decoder0)<-B<-(decoder1)<-C<-(decoder2)<-D

        ft_CB=self.decoder3(ft_CD)
        ft_BA = self.decoder2(ft_BC)        
        x = self.decoder1(ft_AB)

        return ft_CB,ft_BA,x

    def illusion_pass(relu_ft_CD):

        ft_CD_flat=torch.flatten(relu_ft_CD,1)
        ft_CD_flat=self.batch_normal(ft_CD_flat)
        ft_DE = self.fc1(ft_CD_flat)
        relu_DE=F.relu(ft_DE)
        ft_EF=self.fc2(relu_DE)
        relu_EF=F.relu(ft_EF)
        ft_FG=self.fc3(relu_EF)
        output=F.softmax(ft_FG,dim=-1)

        return output




