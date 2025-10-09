import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_and_plotting import classification_accuracy_metric, classification_loss_metric, plot_metrics, recon_pc_loss, eval_pc_accuracy, eval_pc_ill_accuracy
import os
import wandb
from wb_tracker import init_wandb
from add_noise import noisy_img
import torchvision.utils as vutils


def illusion_pc_training(net, trainloader, testloader, pc_train_bool, config):

    if pc_train_bool == "fine_tuning":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(
                net.fc1.parameters()) +
            list(
                net.fc2.parameters()) +
            list(
                net.fc3.parameters()),
            lr=config.lr)
        loss_arr = []
        # In zhoyang's paper finetuning was for only 25 epochs
        for epoch in range(config.epochs):
            running_loss = []
            val_recon_loss = []
            total_correct = np.zeros(config.timesteps + 1)  # ✅ Initialize here
            total_samples = 0  # ✅ Initialize here
            net.train()
            for images, labels, _ in trainloader:
                images, labels = images.to(
                    config.device), labels.to(
                    config.device)
                ft_AB_pc_temp = torch.zeros(
                    config.batch_size, 6, 32, 32).to(
                    config.device)
                ft_BC_pc_temp = torch.zeros(
                    config.batch_size, 16, 16, 16).to(
                    config.device)
                ft_CD_pc_temp = torch.zeros(
                    config.batch_size, 32, 8, 8).to(
                    config.device)
                ft_DE_pc_temp = torch.zeros(
                    config.batch_size, 64, 4, 4).to(
                    config.device)

                ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, ft_FG_ppc_temp, output = net.feedforward_pass(
                    images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp)

                _, predicted = torch.max(output, 1)
                total_correct[0] += (predicted == labels).sum().item()

                # In pc_train.py training loop
                ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp.requires_grad_(True)

                optimizer.zero_grad()
                final_loss = 0
                train_recon_loss = 0
                for i in range(config.timesteps):
                    output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(
                        images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, config.betaset, config.gammaset, config.alphaset, images.size(0))
                    loss = criterion(output, labels)
                    final_loss += loss
                    train_recon_loss += loss_of_layers
                    _, predicted = torch.max(output, 1)
                    total_correct[i + 1] += (predicted == labels).sum().item()

                total_samples += labels.size(0)
                final_loss = final_loss / config.timesteps
                train_recon_loss = train_recon_loss / config.timesteps
                final_loss.backward()
                optimizer.step()
                running_loss.append(final_loss.item())
                val_recon_loss.append(train_recon_loss.item())

                # Clear Batches
                del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, loss_of_layers
                torch.cuda.empty_cache()

            accuracy = [100 * c / total_samples for c in total_correct]
            accuracy = np.array(accuracy)
            train_accuracy = np.mean(accuracy)
            avg_loss = np.mean(running_loss)
            avg_recon_loss = np.mean(val_recon_loss)
            print(
                f"Epoch:{epoch} and AverageLoss:{avg_loss} and Reconstruction Loss:{train_recon_loss}")
            net.eval()
            test_accuracy, test_loss, test_recon_loss = eval_pc_ill_accuracy(
                net, testloader, config, criterion)
            metrics = {
                "Fine_Tuning_With_Classification/train_loss": avg_loss,
                "Fine_Tuning_With_Classification/test_loss": test_loss,
                "Fine_Tuning_With_Classification/Test_Accuracy": test_accuracy,
                "Fine_Tuning_With_Classification/Training_accuracy": train_accuracy,
                "Fine_Tuning_With_Classification/Recon_Training_loss": avg_recon_loss,
                "Fine_Tuning_With_Classification/Recon_Testing_loss": test_recon_loss}
            wandb.log(metrics)
            loss_arr.append(avg_loss)

        return True

    if pc_train_bool == "test":
        # Dictionary to store results per class
        class_results = {
            "Square": {"correct": np.zeros(config.timesteps + 1), "total": 0},
            "Random": {"correct": np.zeros(config.timesteps + 1), "total": 0},
            "All-in": {"correct": np.zeros(config.timesteps + 1), "total": 0},
            "All-out": {"correct": np.zeros(config.timesteps + 1), "total": 0}
        }

        net.eval()

        for images, labels, cls_names in testloader:
            images, labels = images.to(config.device), labels.to(config.device)

            ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
            ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
            ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
            ft_DE_pc_temp = torch.zeros(config.batch_size,64,4,4).to(config.device)

            ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp)

            # Re-enable gradients after feedforward_pass overwrites the tensors
            ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
            ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
            ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
            ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)

            _, predicted = torch.max(output, 1)

            for i, cls_name in enumerate(cls_names):
                if predicted[i].item() == 1:
                    class_results[cls_name]["correct"][0] += 1

            for cls_name in cls_names:
                class_results[cls_name]["total"] += 1

            for t in range(config.timesteps):
                output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, config.betaset, config.gammaset, config.alphaset, images.size(0))

                _, predicted = torch.max(output, 1)

                for i, cls_name in enumerate(cls_names):
                    if predicted[i].item() == 1:
                        class_results[cls_name]["correct"][t + 1] += 1

        for cls_name in ["Square", "Random", "All-in", "All-out"]:
            total = class_results[cls_name]["total"]

            accuracy_over_timesteps = 100 * \
                (class_results[cls_name]["correct"] / total)
            
            print("=================================")
            print(f"Class: {cls_name},Total:{total}")
            for t, acc in enumerate(accuracy_over_timesteps):
                print(f"Timestep {t}:  {acc:2f}%")
            print("\n")

        return None
