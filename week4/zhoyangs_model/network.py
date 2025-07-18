import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ZP_PC_Net(nn.Module):

    def __init__(self):
        super().__init__()
        ##This network does not have any sort of pooling according to the paper and repo which is very weird
        self.encoder1=nn.Conv2d(in_channels=3,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)
        self.encoder2=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)
        self.encoder3=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2)

        self.decoder3=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2,output_padding=1)
        self.decoder2=nn.Conv2d(in_channels=128,out_channels= 128,kernel_size= 5,stride=(2,2),padding=2,output_padding=1)
        self.decoder1=nn.Conv2d(in_channels=128,out_channels= 3,kernel_size= 5,stride=(2,2),padding=2,output_padding=1)

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



    def predictive_coding_pass(self,x,ft_AB,ft_BC,ft_CD,ft_DE,ft_EF,beta,gamma,alpha,batch_size):

        gamma_AB_fwd,gamma_BC_fwd,gamma_CD_fwd=gamma

        beta_AB_bck,beta_BC_bck,beta_CD_bck=beta

        alpha_AB,alpha_BC,alpha_CD,alpha_DE=alpha

        ### First Layer Start ######
        errorB=nn.functional.mse_loss(self.decoder1(ft_AB),x)

        reconstructionB=torch.autograd.grad(errorB,ft_AB,retain_graph=True)[0]

        #This scaling is done by the factor sqrt((k^2/C)) ref: https://proceedings.neurips.cc/paper_files/paper/2021/file/75c58d36157505a600e0695ed0b3a22d-Supplemental.pdf supplement A. The reason why we do this scaling is because by simply dividing it by the number of neurons wouldn't be helpful since not all of them are involved in the receptive field so we should also take into consideration the elements in the receptive and then take their ratio with respect to the the total number of neurons.


        scalingB = np.round(np.sqrt(np.square(32*32*3)/(np.prod(self.encoder1.kernel_size * self.encoder1.in_channels))))

        #The predictive coding has three main terms the forward the backward and the error. Forward is controlled by gamma and backward is controlled by beta and the error gradient is controlled by alpha and the memory term is controlled by 1 - gamma - beta

        ft_AB_pc = gamma_AB_fwd*self.encoder1(x) + (1-gamma_AB_fwd-beta_AB_bck) * ft_AB + beta_AB_bck*self.decoder2(ft_BC)-alpha_AB*scalingB*batch_size*reconstructionB

        ### First Layer End ######

        ### Second Layer Start ######

        errorC=nn.functional.mse_loss(self.decoder2(ft_BC),ft_AB)

        reconstructionC=torch.autograd.grad(errorC,ft_BC,retain_graph=True)[0]

        pooled_ft_AB_pc,indices_AB=self.pool(F.relu(ft_AB_pc))

        scalingC = np.round(np.sqrt(np.square(16*16*6)/(np.prod(self.encoder2.kernel_size * self.encoder2.in_channels))))

        ft_BC_pc = gamma_BC_fwd*self.encoder2(pooled_ft_AB_pc) + (1-gamma_BC_fwd-beta_BC_bck) * ft_BC + beta_BC_bck*self.decoder3(ft_CD) -alpha_BC*scalingC*batch_size*reconstructionC
        
        ### Second Layer End ######

        ### Third Layer Start ######

        errorD=nn.functional.mse_loss(self.decoder3(ft_CD),ft_BC)

        reconstructionD=torch.autograd.grad(errorD,ft_CD,retain_graph=True)[0]

        pooled_ft_BC_pc,indices_BC=self.pool(F.relu(ft_BC_pc))

        scalingD = np.round(np.sqrt(np.square(8*8*16)/(np.prod(self.encoder3.kernel_size * self.encoder3.in_channels))))

        ft_CD_pc= gamma_CD_fwd*self.encoder3(pooled_ft_BC_pc) + (1-gamma_CD_fwd) * ft_CD -alpha_CD*scalingD*batch_size*reconstructionD

        ft_CD_relu=F.relu(ft_CD_pc)

        ### Third Layer End ######

        total_loss=errorB+errorC+errorD
        
        return output,ft_AB_pc,ft_BC_pc,ft_CD_pc,ft_CD_relu,total_loss

    def illusion_pass(relu_ft_CD):
        ft_DE = self.fc1(ft_CD_flat)
        relu_DE=F.relu(ft_DE)
        ft_EF=self.fc2(relu_DE)
        relu_EF=F.relu(ft_EF)
        ft_FG=self.fc3(relu_EF)
        output=F.softmax(ft_FG,dim=-1)

        return output





