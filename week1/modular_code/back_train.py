import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import evaluation_reconstruction,plot_metrics
from config import epochs

def feedback_training(net,ft_AB,ft_BC,ft_CD,ft_DE,output):
    net.train()
    criterion_recon = nn.MSELoss()
    optimizer_bck = optim.SGD(list(net.deconv2_fb.parameters())+list(net.deconv1_fb.parameters())+list(net.fc1_fb.parameters())+list(net.fc2_fb.parameters())+list(net.fc3_fb.parameters()), lr=0.001, momentum=0.9)
    loss_arr = []
    for epoch in range(epochs):
        running_loss = []
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            optimizer_bck.zero_grad()
            ft_AB,ft_BA,ft_BC,ft_CB,ft_CD,ft_DC,ft_DE,ft_ED,xpred = net(images,"backward")
            lossAtoB = criterion_recon(ft_AB, ft_BA)
            lossBtoC = criterion_recon(ft_BC, ft_CB)
            lossCtoD = criterion_recon(ft_CD, ft_DC)
            lossDtoE = criterion_recon(ft_DE, ft_ED)
            loss_input_and_recon = criterion_recon(xpred, images)
            final_loss=lossAtoB+lossBtoC+lossCtoD+lossDtoE+loss_input_and_recon
            final_loss.backward()
            optimizer_bck.step()
            running_loss.append(final_loss.item())

        avg_loss = np.mean(running_loss)
        print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
        loss_arr.append(avg_loss)

    accuracy=evaluation_reconstruction(net,criterion_recon)
    iters = range(1, epochs+1)
    plot_bool=plot_metrics(iters,loss_arr,"backward")
    if plot_bool==True:
        print("Plots Successfully Stored")
    print(f'Backward Connections Accuracy = {accuracy:.2f}%')

    print("Backward Training Succesful")
