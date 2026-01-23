import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from add_noise import noisy_img
from eval_and_plotting import recon_pc_loss, eval_pc_ill_accuracy
import torch.nn.functional as F


def illusion_pc_training(net, trainloader, validationloader, testloader,
                         cifar10_testdata, pc_train_bool, config, metrics_history, model_name):

    if pc_train_bool == "fine_tuning":
        criterion = nn.CrossEntropyLoss()
        
        # NEW: Choose optimizer scope based on config
        optimize_all = getattr(config, 'optimize_all_layers', False)
        
        if optimize_all:
            # Optimize all layers (conv + linear)
            print(f"Optimizer: Training ALL layers (conv + linear)")
            optimizer = optim.Adam(net.parameters(), lr=config.lr)
        else:
            # Optimize only linear layers (default)
            print(f"Optimizer: Training LINEAR layers only (fc1, fc2, fc3)")
            optimizer = optim.Adam(
                list(net.fc1.parameters()) +
                list(net.fc2.parameters()) +
                list(net.fc3.parameters()),
                lr=config.lr
            )

        for epoch in range(config.epochs):
            running_loss = []
            classificaton_losses = []
            reconstruction_losses = []
            val_recon_loss = []
            total_correct = np.zeros(config.timesteps + 1)
            total_samples = 0

            net.train()

            for images, labels, _, _ in trainloader:
                images_orig = images.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()

                # Accumulate loss across all noise levels
                batch_classification_loss = 0
                batch_reconstruction_loss = 0
                num_noise_levels = 0

                for noise in np.arange(0, 0.35, 0.05):
                    images = noisy_img(
                        images_orig.clone(), "gauss", round(noise, 2))

                    # Temporary feature tensors
                    _, _, height, width = images.shape
                    batch_size = images.size(0)

                    ft_AB_pc_temp = torch.zeros(
                                    batch_size, 6, height, width, device=config.device
                                    )

                    ft_BC_pc_temp = torch.zeros(
                                    batch_size, 16, height // 2, width // 2, device=config.device
                                    )

                    ft_CD_pc_temp = torch.zeros(
                                    batch_size, 32, height // 4, width // 4, device=config.device
                                    ) 

                    ft_DE_pc_temp = torch.zeros(
                                    batch_size, 128, height // 8, width // 8, device=config.device
                                    )


                    ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                        images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp)

                    _, predicted = torch.max(output, 1)
                    total_correct[0] += (predicted == labels).sum().item()

                    # Enable gradients
                    ft_AB_pc_temp.requires_grad_(True)
                    ft_BC_pc_temp.requires_grad_(True)
                    ft_CD_pc_temp.requires_grad_(True)
                    ft_DE_pc_temp.requires_grad_(True)


                    noise_classification_loss = 0
                    noise_reconstruction_loss = 0

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
                        classification_loss = criterion(output, labels)
                        noise_classification_loss += classification_loss
                        noise_reconstruction_loss += loss_of_layers.item()

                        _, predicted = torch.max(output, 1)
                        total_correct[i +
                                      1] += (predicted == labels).sum().item()

                    noise_classification_loss = noise_classification_loss / config.timesteps
                    noise_reconstruction_loss = noise_reconstruction_loss / config.timesteps

                    # Accumulate to batch totals
                    batch_classification_loss += noise_classification_loss
                    batch_reconstruction_loss += noise_reconstruction_loss
                    num_noise_levels += 1

                    # Cleanup
                    del ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, ft_EF_pc_temp, loss_of_layers
                    torch.cuda.empty_cache()

                 # Average loss across all noise levels
                avg_classification_loss = batch_classification_loss / num_noise_levels
                avg_reconstruction_loss = batch_reconstruction_loss / num_noise_levels

                # Single backward pass for the entire batch (averaged across
                # noise levels)
                avg_classification_loss.backward()
                optimizer.step()

                # Track total samples across all noise levels
                total_samples += labels.size(0) * num_noise_levels

                running_loss.append(avg_classification_loss.item())
                classificaton_losses.append(avg_classification_loss.item())
                reconstruction_losses.append(avg_reconstruction_loss)

            accuracy = [100 * c / total_samples for c in total_correct]
            train_accuracy = np.mean(accuracy)
            avg_loss = np.mean(running_loss)
            avg_class_loss = np.mean(classificaton_losses)
            avg_recon_loss = np.mean(reconstruction_losses)

            print(f"Epoch {epoch:3d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Class: {avg_class_loss:.4f} | "
                  f"Recon: {avg_recon_loss:.4f} | "
                  f"Acc: {train_accuracy:.2f}%")

            net.eval()
            test_accuracy, test_loss, test_recon_loss = eval_pc_ill_accuracy(
                net, validationloader, config, criterion)

            # Store metrics
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['test_loss'].append(test_loss)
            metrics_history['train_acc'].append(train_accuracy)
            metrics_history['test_acc'].append(test_accuracy)
            metrics_history['illusory_datset_recon_loss'].append(avg_recon_loss)
            metrics_history['cifar10_dataset_recon_loss'].append(test_recon_loss)

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                os.makedirs(
                    f'{config.save_model_path}/classification_models',
                    exist_ok=True)
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

        # ✅ MODIFICATION 1: Update registry and plot metrics at end of training
        from model_tracking import get_tracker
        from eval_and_plotting import plot_training_metrics
        
        tracker = get_tracker()
        tracker.update_status(model_name, "completed")
        tracker.update_metrics(model_name, metrics_history)
        
        print(f"\n{'='*60}")
        print(f"Training completed for {model_name}")
        print(f"Generating individual training plots...")
        print(f"{'='*60}\n")
        
        plot_training_metrics(metrics_history, model_name, config)
        
        print(f"\n✓ Metrics saved to registry")
        print(f"✓ Individual plots generated")
        print(f"{'='*60}\n")

        return metrics_history

    # ============================================================
    # TEST MODE
    # ============================================================
    if pc_train_bool == "test":
        # Get class mapping
        test_dataset = testloader.dataset
        if hasattr(test_dataset, 'dataset'):
            class_to_idx = test_dataset.dataset.class_to_idx
        else:
            class_to_idx = test_dataset.class_to_idx

        print(f"\nTesting model: {model_name}")
        print(f"Class mapping: {class_to_idx}")

        # Initialize results storage
        all_classes = list(class_to_idx.keys())
        class_results = {
            cls: {
                "predictions": [[] for _ in range(config.timesteps + 1)],
                "total": 0
            }
            for cls in all_classes
        }

        net.eval()

        for batch_idx, batch_data in enumerate(testloader):
            images, labels, cls_names, should_see = batch_data
            images_orig = images.to(config.device)
            labels = labels.to(config.device)

            # Process with different noise levels
            for noise in np.arange(0, 0.35, 0.05):
                images = noisy_img(images_orig.clone(), "gauss", round(noise, 2))

                # Initialize feature tensors with actual batch size
                _, _, height, width = images.shape
                batch_size = images.size(0)

                ft_AB_pc_temp = torch.zeros(
                                    batch_size, 6, height, width, device=config.device
                                    )

                ft_BC_pc_temp = torch.zeros(
                                    batch_size, 16, height // 2, width // 2, device=config.device
                                    )

                ft_CD_pc_temp = torch.zeros(
                                    batch_size, 32, height // 4, width // 4, device=config.device
                                    ) 

                ft_DE_pc_temp = torch.zeros(
                                    batch_size, 128, height // 8, width // 8, device=config.device
                                    )

                # Initial feedforward
                ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                    ft_EF_pc_temp, ft_FG_pc_temp, output = net.feedforward_pass(
                        images, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp
                    )

                # Enable gradients for predictive coding
                ft_AB_pc_temp = ft_AB_pc_temp.requires_grad_(True)
                ft_BC_pc_temp = ft_BC_pc_temp.requires_grad_(True)
                ft_CD_pc_temp = ft_CD_pc_temp.requires_grad_(True)
                ft_DE_pc_temp = ft_DE_pc_temp.requires_grad_(True)

                # Get probabilities at timestep 0
                probs = F.softmax(output, dim=1).detach().cpu().numpy()

                # Record timestep 0
                for i, cls_name in enumerate(cls_names):
                    if cls_name in ["all_in", "all_out"]:
                        perceived_class = should_see[i]
                    else:
                        perceived_class = cls_name

                    perceived_idx = class_to_idx[perceived_class]
                    class_results[cls_name]["predictions"][0].append(probs[i, perceived_idx])
                    
                    # Count total samples only once (not per timestep)
                    if noise == 0.0:  # Only count on first noise level
                        class_results[cls_name]["total"] += 1

                # Run predictive coding iterations for timesteps 1 to config.timesteps
                for t in range(config.timesteps):
                    output, ft_AB_pc_temp, ft_BC_pc_temp, ft_CD_pc_temp, ft_DE_pc_temp, \
                        ft_EF_pc_temp, loss_of_layers = net.predictive_coding_pass(
                            images,
                            ft_AB_pc_temp,
                            ft_BC_pc_temp,
                            ft_CD_pc_temp,
                            ft_DE_pc_temp,
                            ft_EF_pc_temp,
                            config.betaset,
                            config.gammaset,
                            config.alphaset,
                            config.batch_size
                        )

                    # Get probabilities at this timestep
                    probs = F.softmax(output, dim=1).detach().cpu().numpy()

                    # Record predictions
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
