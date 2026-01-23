import torch

batch_size = 40
epochs = 20
seed = 2933
lr = 5e-05
momentum = 0.9
classification_datasetpath = "cifar10"
recon_datasetpath = "cifar10"
training_condition = "recon_pc_train"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classification_neurons = 10

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {device}")
    print(f"CUDA device name:{torch.cuda.get_device_name(device) }")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
print(f"Using device: {device}")

timesteps = 10

gammaset = [[0.33, 0.33, 0.33, 0.33]]  # pattern: Uniform
betaset = [[0.33, 0.33, 0.33, 0.33]]  # pattern: Uniform
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "recon_t10_c10_uni_s2933"
noise_type="s&p"
noise_param=0.0

experiment_name = "Testing recon_t10_c10_uni_s2933 with Uniform pattern at 10 timesteps"
load_model_path="/home/ajinkyar/ml_models"
save_model_path="/home/ajinkyar/ml_models"



