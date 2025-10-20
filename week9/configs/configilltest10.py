import torch

batch_size = 40
epochs = 10
seed = 42
lr = 0.00005 
momentum = 0.9
datasetpath = "data/visual_illusion_dataset"
training_condition=None
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {device}")
    print(f"CUDA device name:{torch.cuda.get_device_name(device) }")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
print(f"Using device: {device}")

timesteps = 10

gammaset=[[0.33, 0.33, 0.33, 0.33]]
betaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pc_illusiont10_recon_t1_0"
noise_type="s&p"
noise_param=0.0

experiment_name= "Illusion Training Using 10 timestep with Recon 1 timesteps"
load_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week9/models/illusion_trained_models/illusion_t10"
save_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week9/models/illusion_trained_models/illusion_t10"

illusion_dataset_bool=True

iterations = 20
