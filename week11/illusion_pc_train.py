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
from eval_and_plotting import recon_pc_loss,eval_pc_ill_accuracy


def illusion_pc_training(net, trainloader, validationloader, testloader,
                         cifar10_testdata, pc_train_bool, config, metrics_history, model_name):

    if pc_train_bool == "fine_tuning":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(net.fc1.parameters()) +
            list(net.fc2.parameters()) +
            list(net.fc3.parameters()),
            lr=config.lr
        )

        for epoch in range(config.epochs):
            running_loss = []
            val_recon_loss = []
            total_correct = np.zeros(config.timesteps + 1)
            total_samples = 0

            net.train()

            for images, labels, _, _ in trainloader:
                images_orig = images.to(config.device)
                labels = labels.to(config.device)
                
                for noise in np.arange(0, 0.35, 0.05):
                    images = noisy_img(images_orig.clone(), "gauss", round(noise, 2))

                    # Temporary feature tensors
                    ft_AB_pc_temp = torch.zeros(config.batch_size, 6, 32, 32).to(config.device)
                    ft_BC_pc_temp = torch.zeros(config.batch_size, 16, 16, 16).to(config.device)
                    ft_CD_pc_temp = torch.zeros(config.batch_size, 32, 8, 8).to(config.device)
                    ft_DE_pc_temp = torch.zeros(config.batch_size, 64, 4, 4).to(config.device)

                    ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
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
                        total_correct[i + 1] += (predicted == labels).sum().item()

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

            print(f"Epoch:{epoch} | AvgLoss:{avg_loss:.4f} | TrainAcc:{train_accuracy:.2f}%")

            net.eval()
            test_accuracy, test_loss = eval_pc_ill_accuracy(net, validationloader, config, criterion)
            test_recon_loss = recon_pc_loss(net, cifar10_testdata, config)

            # Store metrics
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['test_loss'].append(test_loss)
            metrics_history['train_acc'].append(train_accuracy)
            metrics_history['test_acc'].append(test_accuracy)
            metrics_history['train_recon_loss'].append(avg_recon_loss)
            metrics_history['test_recon_loss'].append(test_recon_loss)

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                os.makedirs(f'{config.save_model_path}/classification_models', exist_ok=True)
                save_path = f'{config.save_model_path}/classification_models/{model_name}_epoch{epoch}.pth'

                torch.save({
                    "conv1": net.conv1.state_dict(),
                    "conv2": net.conv2.state_dict(),
                    "conv3": net.conv3.state_dict(),
                    "conv4": net.conv4.state_dict(),
                    "fc1": net.fc1.state_dict(),
                    "fc2": net.fc2.state_dict(),
                    "fc3": net.fc3.state_dict(),
                    "deconv1_fb": net.deconv1_fb.state_dict(),
                    "deconv2_fb": net.deconv2_fb.state_dict(),
                    "deconv3_fb": net.deconv3_fb.state_dict(),
                    "deconv4_fb": net.deconv4_fb.state_dict(),
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "test_loss": test_loss,
                    "train_acc": train_accuracy,
                    "test_acc": test_accuracy
                }, save_path)

                print(f"Checkpoint saved: {save_path}")

        return metrics_history

    # ============================================================
    # TEST MODE - COMPLETELY REWRITTEN
    # ============================================================
    if pc_train_bool == "test":
        
        # Get class mapping from testloader
        # testloader.dataset is a Subset, dataset.dataset is the original SquareDataset
        test_dataset = testloader.dataset
        if hasattr(test_dataset, 'dataset'):
            # It's a Subset
            class_to_idx = test_dataset.dataset.class_to_idx
        else:
            # Direct dataset
            class_to_idx = test_dataset.class_to_idx
        
        print(f"\nTesting model: {model_name}")
        print(f"Class mapping: {class_to_idx}")
        
        # Initialize results storage for ALL classes (including all_in, all_out)
        all_classes = list(class_to_idx.keys())
        class_results = {
            cls: {
                "predictions": [[] for _ in range(config.timesteps + 1)],
                "total": 0
            }
            for cls in all_classes
        }
        
        net.eval()
        
        # Don't use torch.no_grad() because predictive_coding_pass needs gradients!
        for batch_idx, batch_data in enumerate(testloader):
            images, labels, cls_names, should_see = batch_data
            
            images_orig = images.to(config.device)
            labels = labels.to(config.device)
            
            # Process with different noise levels
            for noise in np.arange(0, 0.35, 0.05):
                images = noisy_img(images_orig.clone(), "gauss", round(noise, 2))
                
                # Initialize feature tensors with actual batch size
                batch_size = images.size(0)
                ft_AB_pc_temp = torch.zeros(batch_size, 6, 32, 32).to(config.device)
                ft_BC_pc_temp = torch.zeros(batch_size, 16, 16, 16).to(config.device)
                ft_CD_pc_temp = torch.zeros(batch_size, 32, 8, 8).to(config.device)
                ft_DE_pc_temp = torch.zeros(batch_size, 64, 4, 4).to(config.device)
                
                # Initial feedforward
                ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                    images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp
                )
                
                # ✅ CRITICAL: Enable gradients for predictive coding
                ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)
                
                # Get probabilities at timestep 0
                probs = F.softmax(output, dim=1).detach().cpu().numpy()
                
                # Record timestep 0
                for i, cls_name in enumerate(cls_names):
                    # Determine perceived class
                    if cls_name in ["all_in", "all_out"]:
                        perceived_class = should_see[i]
                    else:
                        perceived_class = cls_name
                    
                    perceived_idx = class_to_idx[perceived_class]
                    class_results[cls_name]["predictions"][0].append(probs[i, perceived_idx])
                    
                    # Only count once per image (not per noise level)
                    if noise == 0.0:
                        class_results[cls_name]["total"] += 1
                
                # Predictive coding iterations
                for t in range(config.timesteps):
                    output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                    ft_EF_pc_temp, _ = net.predictive_coding_pass(
                        images,
                        ft_AB_pc_temp,
                        ft_BC_pc_temp,
                        ft_CD_pc_temp,
                        ft_DE_pc_temp,
                        ft_EF_pc_temp,
                        config.betaset,
                        config.gammaset,
                        config.alphaset,
                        batch_size
                    )
                    
                    probs = F.softmax(output, dim=1).detach().cpu().numpy()
                    
                    for i, cls_name in enumerate(cls_names):
                        if cls_name in ["all_in", "all_out"]:
                            perceived_class = should_see[i]
                        else:
                            perceived_class = cls_name
                        
                        perceived_idx = class_to_idx[perceived_class]
                        class_results[cls_name]["predictions"][t + 1].append(probs[i, perceived_idx])
                
                # Cleanup
                del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp
                torch.cuda.empty_cache()
        
        # Print results
        print("\n" + "=" * 80)
        print("ILLUSION TESTING RESULTS")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Timesteps: {config.timesteps}")
        print("=" * 80)
        
        for cls_name in sorted(class_results.keys()):
            if class_results[cls_name]["total"] == 0:
                continue
            
            total = class_results[cls_name]["total"]
            mean_probs = [
                np.mean(p) * 100 if len(p) > 0 else 0.0
                for p in class_results[cls_name]["predictions"]
            ]
            
            print(f"\n{cls_name.upper()}")
            print(f"Total Samples: {total}")
            print("-" * 40)
            for t, prob in enumerate(mean_probs):
                print(f"  Timestep {t:2d}: {prob:6.2f}%")
        
        print("\n" + "=" * 80)
        
        # Save trajectory plot
        from eval_and_plotting import plot_test_trajectory
        plot_test_trajectory(class_results, model_name, config)
        
        return class_results
