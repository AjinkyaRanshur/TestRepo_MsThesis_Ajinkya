import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import evaluation_reconstruction,plot_metrics
from config import epochs,seed
import os

def feedback_training(net,trainloader,testloader,lr,momentum,save_dir):
    
    net.train()

    forward_params = [
    net.conv1, net.conv2, net.fc1, net.fc2, net.fc3]

    for module in forward_params:
        for param in module.parameters():
            param.requires_grad = False

    feedback_params = [
        net.fc3_fb, net.fc2_fb, net.fc1_fb, 
        net.deconv2_fb, net.deconv1_fb
    ]

    for module in feedback_params:
        for param in module.parameters():
            param.requires_grad = True

    criterion_recon = nn.functional.mse_loss
    optimizer_bck = optim.SGD(list(net.deconv2_fb.parameters())+list(net.deconv1_fb.parameters())+list(net.fc1_fb.parameters())+list(net.fc2_fb.parameters())+list(net.fc3_fb.parameters()), lr=lr, momentum=momentum)
    loss_arr = []
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_bck.zero_grad()
            ft_AB,ft_BC,ft_CD,ft_DE,output,indices_AB,indices_BC=net.feedforward_pass(images)
            ft_BA,ft_CB,ft_DC,ft_ED,xpred = net.feedback_pass(output,indices_AB,indices_BC,ft_AB,ft_BC,ft_CD,ft_DE)
            # # Flatten ft_BC for comparison with ft_CB (which is also flattened in feedback)
            # ft_BC = ft_BC.view(ft_BC.size(0), -1)
            # ft_CB = ft_CB.view(ft_CB.size(0), -1)
            lossAtoB = criterion_recon(ft_AB, ft_BA)
            lossBtoC = criterion_recon(ft_BC, ft_CB)
            lossCtoD = criterion_recon(ft_CD, ft_DC)
            lossDtoE = criterion_recon(ft_DE, ft_ED)
            loss_input_and_recon = criterion_recon(xpred, images)
            #print("lossAtoB",lossAtoB)
            #print("lossBtoC",lossBtoC)
            #print("lossCtoD",lossCtoD)
            #print("lossDtoE",lossDtoE)
            #print("loss_input_and_recon",loss_input_and_recon)
            final_loss=lossAtoB+lossBtoC+lossCtoD+lossDtoE+loss_input_and_recon
            final_loss=final_loss/5.0
            final_loss.backward()
            optimizer_bck.step()
            running_loss.append(final_loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_reconstruction(net,testloader)
    iters = range(1, epochs+1)
    plot_bool=plot_metrics(iters,loss_arr,"backward",save_dir)
    if plot_bool==True:
        print("Plots Successfully Stored")

    file_path=os.path.join(save_dir,f"Accuracy_Stats_{seed}.txt")
    with open(file_path,"a") as f:
        f.write(f"Backward Connection Accuracy= {accuracy:.2f} with seed ={seed}\n")
    #print(f'Backward Connections Accuracy = {accuracy:.2f}%')

    print("Backward Training Succesful")
