import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import eval_pc_accuracy,recon_pc_loss,plot_metrics
import os
import wandb
from wb_tracker import init_wandb
from add_noise import noisy_img
import torchvision.utils as vutils

def class_pc_training(net,trainloader,testloader,pc_train_bool,config):

    if pc_train_bool=="train":
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.SGD(net.parameters(),lr=config.lr,momentum=config.momentum)
        loss_arr=[]
        for epoch in range(config.epochs):
            running_loss=[]
            total_correct = np.zeros(config.timesteps + 1)  # ✅ Initialize here
            total_samples = 0  # ✅ Initialize here
            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(config.device),labels.to(config.device)
                
                ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
                ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
                ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
                ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

                _,predicted=torch.max(output,1)
                total_correct[0]+=(predicted==labels).sum().item()

                # In pc_train.py training loop       
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)
                ft_EF_pc_temp.requires_grad_(True)
                optimizer.zero_grad()
                final_loss=0
                for i in range(config.timesteps):
                    output,ft_AB_pc,ft_BC_pc,ft_CD_pc,ft_DE_pc,ft_EF_pc,_=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
                    loss=criterion(output,labels)
                    final_loss+=loss
                    _,predicted=torch.max(output,1)
                    total_correct[i+1]+=(predicted==labels).sum().item()

                total_samples+=labels.size(0)

                final_loss=final_loss/config.timesteps
                final_loss.backward()
                optimizer.step()
                running_loss.append(final_loss.item())
            
            accuracy=[100 * c /total_samples for c in total_correct]
            accuracy=np.array(accuracy)
            train_accuracy=np.mean(accuracy)
            avg_loss=np.mean(running_loss)
            print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
            test_accuracy,test_loss,_=eval_pc_accuracy(net,testloader,config,criterion)
            metrics={"Classification_with_predictive_coding/train_loss":avg_loss,"Classification_with_predictive_coding/test_loss":test_loss,"Classification_with_predictive_coding/Testing_accuracy":test_accuracy,"Classification_with_predictive_coding/Training_accuracy":train_accuracy }
            wandb.log(metrics)
            loss_arr.append(avg_loss)
        iters = range(1, config.epochs+1)

        return True



    if pc_train_bool=="test":

        forward_params = [
        net.conv1, net.conv2,net.conv3,net.conv4,net.fc1, net.fc2]

        for module in forward_params:
            for param in module.parameters():
                param.requires_grad = False

        feedback_params = [
            net.deconv4_fb,net.deconv3_fb, net.fc2_fb, net.fc1_fb, 
            net.deconv2_fb, net.deconv1_fb
        ]
        for module in feedback_params:
            for param in module.parameters():
                param.requires_grad = False

        
        total_correct=np.zeros(config.timesteps+1)
        total_samples=0
        
        for batch_id,batch in enumerate(trainloader):
            
            images,labels=batch
            #Adding noise to the image
            images=noisy_img(images,config.noise_type,config.noise_param)
            if batch_id == 0:
                classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
                wandb_images=[]
                if batch_id == 0:
                    # Clamp or normalize images to [0, 1] for correct visualization
                    images = images.clone().detach()
                    images = images - images.min()
                    images = images / images.max()
                grid = vutils.make_grid(images[:8], nrow=4, padding=2, normalize=False)
                
            # Move data to the same device as the model
            images, labels = images.to(config.device), labels.to(config.device)
            ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32)
            ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16)
            ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8)
            ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4)

            ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

            # Re-enable gradients after feedforward_pass overwrites the tensors
            ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
            ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)
            ft_EF_pc_temp = ft_EF_pc_temp.requires_grad_(True)

            _,predicted=torch.max(output,1)
            total_correct[0]+=(predicted==labels).sum().item()

            for i in range(config.timesteps):
                output,ft_AB_pc,ft_BC_pc,ft_CD_pc,ft_DE_pc,ft_EF_pc,_=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
                _,predicted=torch.max(output,1)
                total_correct[i+1]+=(predicted==labels).sum().item()

            total_samples+=labels.size(0)


        accuracy=[100 * c /total_samples for c in total_correct]
        #iters=range(0,timesteps+1)

        print("Accuracy at each timestep:")
        for i, acc in enumerate(accuracy):
            print(f"Timestep {i}: {acc:.2f}%")
            wandb.log({"Timestep":i,"Accuracy":acc,"chance_level":10.00})


        return accuracy

