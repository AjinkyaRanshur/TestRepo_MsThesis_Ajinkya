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
from eval_and_plotting



def illusion_pc_training(net, trainloader, testloader, pc_train_bool, config,metrics_history):

    if pc_train_bool == "fine_tuning":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(net.fc1.parameters()) +
            list(net.fc2.parameters()) +
            list(net.fc3.parameters()),
            lr=config.lr
        )

        loss_arr = []

        # Fine-tuning (as in Zhoyang's paper, ~25 epochs)
        for epoch in range(config.epochs):
            running_loss = []
            val_recon_loss = []
            total_correct = np.zeros(config.timesteps + 1)
            total_samples = 0

            net.train()

            for images, labels, _ in trainloader:
                images_orig = images.to(config.device)
                labels = labels.to(config.device)
                for noise in np.arange(0, 0.35, 0.05):
                    images = noisy_img(images_orig.clone(), "gauss", round(noise, 2))

                    # Temporary feature tensors
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

                    # Enable gradients
                    ft_AB_pc_temp.requires_grad_(True)
                    ft_BC_pc_temp.requires_grad_(True)
                    ft_CD_pc_temp.requires_grad_(True)
                    ft_DE_pc_temp.requires_grad_(True)

                    optimizer.zero_grad()
                    final_loss = 0
                    train_recon_loss = 0

                    for i in range(config.timesteps):
                        output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(
                            images,
                            ft_AB_pc_temp,
                            ft_BC_pc_temp,
                            ft_CD_pc_temp,
                            ft_DE_pc_temp,
                            ft_EF_pc_temp,
                            config.betaset,
                            config.gammaset,
                            config.alphaset,
                            images.size(0),
                        )
                        loss = criterion(output, labels)
                        final_loss += loss
                        train_recon_loss += loss_of_layers
                        _, predicted = torch.max(output, 1)
                        total_correct[i +
                                      1] += (predicted == labels).sum().item()

                    total_samples += labels.size(0)
                    final_loss = final_loss / config.timesteps
                    train_recon_loss = train_recon_loss / config.timesteps

                    final_loss.backward()
                    optimizer.step()

                    running_loss.append(final_loss.item())
                    val_recon_loss.append(train_recon_loss.item())

                    # Cleanup
                    del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, loss_of_layers
                    torch.cuda.empty_cache()

            accuracy = [100 * c / total_samples for c in total_correct]
            train_accuracy = np.mean(accuracy)
            avg_loss = np.mean(running_loss)
            avg_recon_loss = np.mean(val_recon_loss)

            print(
                f"Epoch:{epoch} | AvgLoss:{avg_loss:.4f} | ReconLoss:{avg_recon_loss:.4f}")

            net.eval()
            test_accuracy, test_loss, test_recon_loss = eval_pc_ill_accuracy(
                net, testloader, config, criterion)

            # ✅ ADD THIS: Store metrics
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['test_loss'].append(test_loss)
            metrics_history['train_acc'].append(train_accuracy)
            metrics_history['test_acc'].append(test_accuracy)
            metrics_history['train_recon_loss'].append(avg_recon_loss)
            metrics_history['test_recon_loss'].append(test_recon_loss)

            metrics = {
                "Fine_Tuning/train_loss": avg_loss,
                "Fine_Tuning/test_loss": test_loss,
                "Fine_Tuning/test_accuracy": test_accuracy,
                "Fine_Tuning/train_accuracy": train_accuracy,
                "Fine_Tuning/recon_train_loss": avg_recon_loss,
                "Fine_Tuning/recon_test_loss": test_recon_loss,
            }


        return metrics_history

    # ------------------------------------------------------------
    # TEST MODE
    # ------------------------------------------------------------
    if pc_train_bool == "test":
        # Store results per class
        class_results = {
            "Square": {"predictions": [[] for _ in range(config.timesteps + 1)], "total": 0},
            "Random": {"predictions": [[] for _ in range(config.timesteps + 1)], "total": 0},
            "All-in": {"predictions": [[] for _ in range(config.timesteps + 1)], "total": 0},
            "All-out": {"predictions": [[] for _ in range(config.timesteps + 1)], "total": 0},
        }

        net.eval()

        for images, labels, cls_names in testloader:
            images_orig, labels = images.to(config.device), labels.to(config.device)

            for cls_name in cls_names:
                class_results[cls_name]["total"] += 1

            for noise in np.arange(0, 0.35, 0.05):
                    images = noisy_img(images_orig.clone(), "gauss", round(noise, 2))

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

                    ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                        images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp
                    )

                    # Re-enable gradients
                    ft_AB_pc_temp.requires_grad_(True)
                    ft_BC_pc_temp.requires_grad_(True)
                    ft_CD_pc_temp.requires_grad_(True)
                    ft_DE_pc_temp.requires_grad_(True)

                    probs = F.softmax(output, dim=1)
                    square_probs = probs[:, 0].detach().cpu().numpy()

                    for i, cls_name in enumerate(cls_names):
                        class_results[cls_name]["predictions"][0].append(
                            square_probs[i])


                    # Predictive coding timesteps
                    for t in range(config.timesteps):
                        output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, _ = net.predictive_coding_pass(
                            images,
                            ft_AB_pc_temp,
                            ft_BC_pc_temp,
                            ft_CD_pc_temp,
                            ft_DE_pc_temp,
                            ft_EF_pc_temp,
                            config.betaset,
                            config.gammaset,
                            config.alphaset,
                            images.size(0),
                        )

                        probs = F.softmax(output, dim=1)
                        square_probs = probs[:, 0].detach().cpu().numpy()

                        for i, cls_name in enumerate(cls_names):
                            class_results[cls_name]["predictions"][t +
                                                                1].append(square_probs[i])

        # Compute mean probability per timestep for each class
        for cls_name in ["Square", "Random", "All-in", "All-out"]:
            total = class_results[cls_name]["total"]
            mean_probs = [
                np.mean(p) *
                100 for p in class_results[cls_name]["predictions"]]

            print("=================================")
            print(f"Class: {cls_name}, Total Samples: {total}")
            for t, acc in enumerate(mean_probs):
                print(f"Timestep {t}: {acc:.2f}%")
            print("\n")

        return class_results

