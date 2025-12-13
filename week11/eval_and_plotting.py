import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from config import seed, device, batch_size, epochs, lr, momentum, timesteps,training_condition
import os
from PIL import Image
import json
from add_noise import noisy_img



def recon_pc_loss(net,dataloader,config):
    
    total_loss=[]
    test_loss=0
    for batch_idx,batch in enumerate(dataloader):
        images,labels=batch
        images,labels=images.to(config.device),labels.to(config.device)

        ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
        ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
        ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
        ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

        ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass_no_dense(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

        # Re-enable gradients after feedforward_pass overwrites the tensors
        # Only enable gradients for the specific tensors that need them
        ft_AB_pc_temp.requires_grad_(True)
        ft_BC_pc_temp.requires_grad_(True)
        ft_CD_pc_temp.requires_grad_(True)
        ft_DE_pc_temp.requires_grad_(True)

        final_loss=0

        for i in range(config.timesteps):
            #print("Timestep",i)
            #print("Batch Id",batch_idx)
            ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
            final_loss+=loss_of_layers

        final_loss=final_loss/config.timesteps
        total_loss.append(final_loss.item())
        # Clear batch tensors
        del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,loss_of_layers,final_loss
        torch.cuda.empty_cache()

    test_loss=np.mean(total_loss)

    return test_loss















