import torch

batch_size = 40
epochs = 10
seed = 42
lr = 0.00005 
momentum = 0.9
datasetpath = '/home/ajinkya/datasets'
training_condition="recon_comparison"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {device}")
    print(f"CUDA device name:{torch.cuda.get_device_name(device) }")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
print(f"Using device: {device}")

timesteps = 1

gammaset=[[0.33, 0.33, 0.33, 0.33]]
betaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pc_recon_train_t1"
noise_type="s&p"
noise_param=0.0

experiment_name= "Reconstruction Comparison"
load_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/recon_models/timesteps1"
save_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/recon_models/timesteps1"
illusion_dataset_bool=False

