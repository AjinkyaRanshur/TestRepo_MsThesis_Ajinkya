import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import recon_loss
import wandb
import os

def feedback_training(net, trainloader, testloader, lr, momentum, save_dir,epochs,seed,device,batch_size):
    
    net.train()

    forward_params = [
    net.conv1, net.conv2, net.conv3,net.conv4,net.fc1, net.fc2]

    for module in forward_params:
        for param in module.parameters():
            param.requires_grad = False

    feedback_params = [
        net.fc2_fb, net.fc1_fb,net.deconv4_fb,net.deconv3_fb, 
        net.deconv2_fb, net.deconv1_fb
    ]

    for module in feedback_params:
        for param in module.parameters():
            param.requires_grad = True

    criterion_recon = nn.functional.mse_loss
    optimizer_bck = optim.SGD(list(net.deconv4_fb.parameters())+list(net.deconv3_fb.parameters())+list(net.deconv2_fb.parameters())+list(net.deconv1_fb.parameters())+list(net.fc2_fb.parameters())+list(net.fc1_fb.parameters()), lr=lr, momentum=momentum)
    loss_arr = []
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            ft_AB = torch.zeros(batch_size, 6, 32, 32)
            ft_BC = torch.zeros(batch_size, 16, 16, 16)
            ft_CD = torch.zeros(batch_size, 32, 8, 8)
            ft_DE = torch.zeros(batch_size,64,4,4)
            images, labels = batch
            images,labels=images.to(device),labels.to(device)
            optimizer_bck.zero_grad()
            ft_AB,ft_BC,ft_CD,ft_DE,ft_EF,output=net.feedforward_pass(images,ft_AB,ft_BC,ft_CD,ft_DE)
            ft_BA,ft_CB,ft_DC,ft_ED,ft_FE,xpred = net.feedback_pass(output,ft_AB,ft_BC,ft_CD,ft_DE,ft_EF)
            # # Flatten ft_BC for comparison with ft_CB (which is also flattened in feedback)
            lossAtoB = criterion_recon(ft_AB, ft_BA)
            lossBtoC = criterion_recon(ft_BC, ft_CB)
            #print(ft_CD.size(),ft_DC.size())
            lossCtoD = criterion_recon(ft_CD,ft_DC)
            lossDtoE = criterion_recon(ft_DE,ft_ED)
            lossEtoF = criterion_recon(ft_EF,ft_FE)
            loss_input_and_recon = criterion_recon(xpred, images)
            final_loss=lossAtoB+lossBtoC+lossCtoD+lossDtoE+loss_input_and_recon+lossEtoF
            final_loss=final_loss/6.0
            final_loss.backward()
            optimizer_bck.step()
            running_loss.append(final_loss.item())

        avg_loss = np.mean(running_loss)
        testloss=recon_loss(net,testloader,batch_size,device,criterion)
        print(f"Epoch:{epoch} and TrainLoss:{avg_loss}")
        metrics={"Reconstruction_Model/Recon_train_loss":avg_loss,"Reconstruction_model/Recon_test_loss":testloss}
        wandb.log(metrics)
        loss_arr.append(avg_loss)

    #accuracy=evaluation_reconstruction(net,testloader)
    #iters = range(1, epochs+1)
    #plot_bool=plot_metrics(iters,loss_arr,save_dir,"Number of Epochs","Average Loss","FeedBack Training Loss","averageloss_vs_epoch_backward",seed)
    #if plot_bool==True:
    #    print("Plots Successfully Stored")

    print("Backward Training Succesful")

