import wandb
from config import batch_size,epochs,lr,momentum,seed,device,training_condition,load_model,save_model,timesteps,gammaset,betaset,alphaset,datasetpath

wandb.login(key="f523ba1b9f976511455de2b9e78f37eaf45c7ab9")

def init_wandb(name):
    
    wandb.init(
            project="Test WandB",
            name=name,
            config={
            "learning_rate":lr,
            "architecture":"CNN",
            "dataset":"CIFAR-10",
            "epochs":epochs,
            "Batch_size":batch_size,
            "momentum":momentum,
            "timesteps":timesteps,
            "training_condition":training_condition,
            "Training The Model":save_model,
            "Testing The Model":load_model,
                }
            )


    return None



