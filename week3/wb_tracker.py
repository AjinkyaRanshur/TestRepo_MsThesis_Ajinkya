import wandb
import os
#from config import batch_size, epochs, lr, momentum, seed, device, training_condition, load_model, save_model, timesteps, gammaset, betaset, alphaset, datasetpath

wandb.login(key="f523ba1b9f976511455de2b9e78f37eaf45c7ab9")
os.environ["WANDB_MODE"] = "online"


def init_wandb(batch_size, epochs, lr, momentum, seed, device, training_condition, load_model, save_model, timesteps, gammaset, betaset, alphaset, datasetpath,name):

    wandb.init(
        project="Test_Sweep",
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
            "Training The Model": save_model,
            "Testing The Model": load_model,
        }
    )

    print("✓ WandB initialized in Online mode")
    #print(f"📁 Logs will be saved to: {wandb.run.dir}")

    return None
