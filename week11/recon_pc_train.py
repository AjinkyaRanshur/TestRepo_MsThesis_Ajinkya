import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from add_noise import noisy_img
import torchvision.utils as vutils
from eval_and_plotting import recon_pc_loss

def recon_pc_training(net,trainloader,testloader,pc_train_bool,config,metrics_history,model_name):

    if pc_train_bool=="train":
        criterion=nn.CrossEntropyLoss()
        #metrics_history = {'train_loss': [], 'test_loss': []}
        optimizer = optim.Adam(list(net.deconv4_fb.parameters())+list(net.deconv3_fb.parameters())+list(net.deconv2_fb.parameters())+list(net.deconv1_fb.parameters())+ list(net.conv1.parameters())+list(net.conv2.parameters())+list(net.conv3.parameters())+list(net.conv4.parameters()), lr=config.lr)
        loss_arr=[]
        iteration_index=0
        for epoch in range(config.epochs):
            running_loss=[]
            val_loss=[]            
            ##Switch to training Mode
            net.train()
            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(config.device),labels.to(config.device)
                ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
                ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
                ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
                ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp = net.feedforward_pass_no_dense(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)

                # In pc_train.py training loop       
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)

                optimizer.zero_grad()
                
                final_loss=0
                for i in range(config.timesteps):
                    ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,loss_of_layers=net.recon_predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
                    final_loss+=loss_of_layers

                final_loss=final_loss/config.timesteps
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
            test_loss=recon_pc_loss(net,testloader,config)
            print(f"Test Loss: {test_loss:.4f}")
            
            # ✅ Store metrics
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['test_loss'].append(test_loss)
            if epoch % 10 == 0:
               iteration_index= iteration_index +1
               save_path = f'{config.save_model_path}/recon_models/{model_name}_{iteration_index}.pth'
               torch.save({
                   "conv1": net.conv1.state_dict(),
                   "conv2": net.conv2.state_dict(),
                   "conv3": net.conv3.state_dict(),
                   "conv4": net.conv4.state_dict(),
                   "deconv1_fb": net.deconv1_fb.state_dict(),
                   "deconv2_fb": net.deconv2_fb.state_dict(),
                   "deconv3_fb": net.deconv3_fb.state_dict(),
                   "deconv4_fb": net.deconv4_fb.state_dict(),
               }, save_path)

               print(f"Model Saved Successfully to: {save_path}")
               print("Model Save Sucessfully")
               
               from model_tracking import get_tracker
               tracker = get_tracker()
               tracker.set_checkpoint_path(model_name,save_path)

        return metrics_history

    if pc_train_bool=="fine_tuning":
        criterion=nn.CrossEntropyLoss()
        #optimizer=optim.SGD(net.parameters(),lr=config.lr,momentum=config.momentum)
        optimizer= optim.Adam(list(net.fc1.parameters())+list(net.fc2.parameters())+list(net.fc3.parameters()), lr=config.lr)
        loss_arr=[]
        ##In zhoyang's paper finetuning was for only 25 epochs
        for epoch in range(config.epochs):
            running_loss=[]
            val_recon_loss=[]
            total_correct = np.zeros(config.timesteps + 1)  # ✅ Initialize here
            total_samples = 0  # ✅ Initialize here
            net.train()
            for batch_idx,batch in enumerate(trainloader):
                images,labels=batch
                images,labels=images.to(config.device),labels.to(config.device)

                ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
                ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
                ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
                ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

                ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,ft_FG_ppc_temp,output = net.feedforward_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp)
                
                _,predicted=torch.max(output,1)
                total_correct[0]+=(predicted==labels).sum().item()

                # In pc_train.py training loop
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)


                optimizer.zero_grad()
                final_loss=0
                train_recon_loss=0
                for i in range(config.timesteps):
                    output,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,loss_of_layers=net.predictive_coding_pass(images,ft_AB_pc_temp,ft_BC_pc_temp,ft_CD_pc_temp,ft_DE_pc_temp,ft_EF_pc_temp,config.betaset,config.gammaset,config.alphaset,images.size(0))
                    loss=criterion(output,labels)
                    #print("+++++++++++++++")
                    #print("Loss of Layers")
                    #print(loss_of_layers)
                    final_loss+=loss
                    train_recon_loss+=loss_of_layers
                    _,predicted=torch.max(output,1)
                    total_correct[i+1]+=(predicted==labels).sum().item()

                total_samples+=labels.size(0)
                final_loss=final_loss/config.timesteps
                train_recon_loss=train_recon_loss/config.timesteps
                final_loss.backward()
                optimizer.step()
                running_loss.append(final_loss.item())
                val_recon_loss.append(train_recon_loss.item())

                #Clear Batches
                del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp,ft_EF_pc_temp,loss_of_layers
                torch.cuda.empty_cache()
            
            accuracy=[100 * c /total_samples for c in total_correct]
            accuracy=np.array(accuracy)
            train_accuracy=np.mean(accuracy)
            avg_loss=np.mean(running_loss)
            avg_recon_loss=np.mean(val_recon_loss)
            print(f"Epoch:{epoch} and AverageLoss:{avg_loss} and Reconstruction Loss:{train_recon_loss}")
            net.eval()
            test_accuracy,test_loss,test_recon_loss=eval_pc_accuracy(net,testloader,config,criterion)
            metrics={"Fine_Tuning_With_Classification/train_loss":avg_loss,"Fine_Tuning_With_Classification/test_loss":test_loss,"Fine_Tuning_With_Classification/Test_Accuracy":test_accuracy,"Fine_Tuning_With_Classification/Training_accuracy":train_accuracy,"Fine_Tuning_With_Classification/Recon_Training_loss":avg_recon_loss,"Fine_Tuning_With_Classification/Recon_Testing_loss":test_recon_loss }
            loss_arr.append(avg_loss)

        return True

