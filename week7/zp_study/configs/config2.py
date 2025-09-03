import torch

batch_size = 128
epochs = 10
seed = 42
lr = 0.001 
momentum = 0.9
datasetpath = '/home/ajinkya/datasets'
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

gammaset=[[0.25, 0.25, 0.25, 0.25]]
betaset = [[0.50, 0.50, 0.50, 0.50]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pc_class_train_balanced"
noise_type="s&p"
noise_param=0.0
experiment_name= "high backward training"
load_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week7/zp_study/models/testing_models"
save_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week7/zp_study/models/testing_models"


