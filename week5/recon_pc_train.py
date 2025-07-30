import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import classification_accuracy_metric,classification_loss_metric,plot_metrics,recon_pc_loss,eval_pc_accuracy
import os
import wandb
from wb_tracker import init_wandb
from add_noise import noisy_img
import torchvision.utils as vutils

def recon_pc_training(net,trainloader,testloader,lr,momentum,save_dir,gamma,beta,alpha,pc_train_bool,epochs,seed,device,timesteps,batch_size,noise_type,noise_param):

    if pc_train_bool=="train":
        criterion=nn.CrossEntropyLoss()
        optimizer = optim.SGD(list(net.deconv4_fb.parameters())+list(net.deconv3_fb.parameters())+list(net.deconv2_fb.parameters())+list(net.deconv1_fb.parameters())+ list(net.conv1.parameters())+list(net.conv2.parameters())+list(net.conv3.parameters())+list(net.conv4.parameters()), lr=lr, momentum=momentum)

        loss_arr=[]
        for epoch in range(epochs):
            running_loss=[]
            val_loss=[]            
            ##Switch to training Mode
            net.train()
            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(device),labels.to(device)
                
                ft_AB_pc_temp = torch.zeros(batch_size, 6, 32, 32).to(device)
                ft_BC_pc_temp = torch.zeros(batch_size, 16, 16, 16).to(device)
                ft_CD_pc_temp = torch.zeros(batch_size, 32, 8, 8).to(device)
                ft_DE_pc_temp = torch.zeros(batch_size,64,4,4).to(device)

                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

                # In pc_train.py training loop       
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)

                optimizer.zero_grad()
                
                final_loss=0
                for i in range(timesteps):
                    ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,beta,gamma,alpha,images.size(0))
                    final_loss+=loss_of_layers

                final_loss=final_loss/timesteps
                final_loss.backward()
                optimizer.step()
                running_loss.append(final_loss.item())
                
                del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,loss_of_layers,final_loss
                torch.cuda.empty_cache()


            avg_loss=np.mean(running_loss)

            print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
            test_loss=0
            ##Set to eval to avoid memory constraints
            net.eval()
            for batch_idx,batch in enumerate(testloader):
                images,labels=batch
                images,labels=images.to(device),labels.to(device)

                ft_AB_pc_temp = torch.zeros(batch_size, 6, 32, 32).to(device)
                ft_BC_pc_temp = torch.zeros(batch_size, 16, 16, 16).to(device)
                ft_CD_pc_temp = torch.zeros(batch_size, 32, 8, 8).to(device)
                ft_DE_pc_temp = torch.zeros(batch_size,64,4,4).to(device)

                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

                # Re-enable gradients after feedforward_pass overwrites the tensors
                # Only enable gradients for the specific tensors that need them
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)

                final_loss=0
                #ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,beta,gamma,alpha,images.size(0))


                for i in range(timesteps):
                    #print("Timestep",i)
                    #print("Batch Id",batch_idx)
                    ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,beta,gamma,alpha,images.size(0))
                    final_loss+=loss_of_layers

                final_loss=final_loss/timesteps
                val_loss.append(final_loss.item())
                #test_loss+=loss_of_layers
                # Clear batch tensors
                del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,loss_of_layers
                torch.cuda.empty_cache()
        
            test_loss=np.mean(val_loss)
            print("Test Loss",test_loss)
            #test_loss=recon_pc_loss(net,testloader,batch_size,beta,gamma,alpha,device,criterion,timesteps)
            metrics={"Reconstruction_with_Predictive_Coding/train_loss":avg_loss,"Reconstruction_with_Predictive_Coding/test_loss":test_loss}
            wandb.log(metrics)
            loss_arr.append(avg_loss)

        return True

    if pc_train_bool=="fine_tuning":
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.SGD(net.parameters(),lr=lr,momentum=momentum)
        loss_arr=[]
        ##In zhoyang's paper finetuning was for only 25 epochs
        for epoch in range(epochs):
            running_loss=[]
            total_correct = np.zeros(timesteps + 1)  # ✅ Initialize here
            total_samples = 0  # ✅ Initialize here

            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(device),labels.to(device)

                ft_AB_pc_temp = torch.zeros(batch_size, 6, 32, 32).to(device)
                ft_BC_pc_temp = torch.zeros(batch_size, 16, 16, 16).to(device)
                ft_CD_pc_temp = torch.zeros(batch_size, 32, 8, 8).to(device)
                ft_DE_pc_temp = torch.zeros(batch_size,64,4,4).to(device)

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

                for i in range(timesteps):
                    output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,beta,gamma,alpha,images.size(0))
                    loss=criterion(output,labels)
                    final_loss+=loss
                    _,predicted=torch.max(output,1)
                    total_correct[i+1]+=(predicted==labels).sum().item()

                total_samples+=labels.size(0)


                final_loss=final_loss/timesteps
                final_loss.backward(retain_graph=True)
                optimizer.step()
                running_loss.append(final_loss.item())
            
            accuracy=[100 * c /total_samples for c in total_correct]
            accuracy=np.array(accuracy)
            train_accuracy=np.mean(accuracy)
            avg_loss=np.mean(running_loss)
            print(f"Epoch:{epoch} and AverageLoss:{avg_loss}")
            test_loss=recon_pc_loss(net,testloader,batch_size,beta,gamma,alpha,device,criterion,timesteps)
            test_accuracy=eval_pc_accuracy(net,testloader,batch_size,beta,gamma,alpha,noise_type,noise_param,device,timesteps)
            metrics={"Reconstruction_with_Predictive_Coding/fine_tuned_train_loss":avg_loss,"Reconstruction_with_Predictive_Coding/fine_tuned_test_loss":test_loss,"Reconstruction_with_Predictive_Coding/fine_tuned_Testing_accuracy":test_accuracy,"Reconstruction_with_Predictive_Coding/fine_tuned_Training_accuracy":train_accuracy }
            wandb.log(metrics)
            loss_arr.append(avg_loss)

        return True



