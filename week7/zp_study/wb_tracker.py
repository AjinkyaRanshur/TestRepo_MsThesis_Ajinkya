import os
os.environ["WANDB_API_KEY"] = "f523ba1b9f976511455de2b9e78f37eaf45c7ab9"

import wandb
wandb.login()


def init_wandb(batch_size, epochs, lr, momentum, seed, device, training_condition,timesteps, gammaset, betaset, alphaset, datasetpath,name,noise_type,noise_param,model_name):

    wandb.init(
        project="random_dumping_projects",
        name=name,
        mode="online",
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "epochs": epochs,
            "Batch_size": batch_size,
            "momentum": momentum,
            "timesteps": timesteps,
            "training_condition": training_condition,
            "Model_Name":model_name,
            "Gamma_layer0":gammaset[0][0],
            "Gamma_layer1":gammaset[0][1],
            "Gamma_layer2":gammaset[0][2],
            "Gamma_layer3":gammaset[0][3],
            "Beta_layer0":betaset[0][0],
            "Beta_layer1":betaset[0][1],
            "Beta_layer2":betaset[0][2],
            "Beta_layer3":betaset[0][3],
            "Alpha_layer0":alphaset[0][0],
            "Alpha_layer1":alphaset[0][1],
            "Alpha_layer2":alphaset[0][2],
            "Alpha_layer3":alphaset[0][3],
            "noise_type":noise_type,
            "noise_param":noise_param

        }
    )

    print("✓ WandB initialized in Online mode")
    #print(f"📁 Logs will be saved to: {wandb.run.dir}")

    return None

