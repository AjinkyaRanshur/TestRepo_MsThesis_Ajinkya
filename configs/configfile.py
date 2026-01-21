import torch

batch_size = 40
epochs = 200
seed = 42
lr = 0.00005 
momentum = 0.9
classification_datasetpath="customillusion"
recon_datasetpath="cifar10"
training_condition = "recon_pc_train"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classification_neurons=10

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {device}")
    print(f"CUDA device name:{torch.cuda.get_device_name(device) }")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
print(f"Using device: {device}")

timesteps = 10

gammaset = [[0.33, 0.33, 0.33, 0.33]]  # pattern: Grid_g0.43_b0.43
betaset = [[0.33, 0.33, 0.33, 0.33]]  # pattern: Grid_g0.43_b0.43
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pc_recon_t10_uniform"
noise_type="s&p"
noise_param=0.0

experiment_name = "Testing pc_recon_t10_class_t10_gamma_increasing with Grid_g0.43_b0.43 pattern at 100 timesteps"
load_model_path="/home/ajinkyar/ml_models"
save_model_path="/home/ajinkyar/ml_models"



