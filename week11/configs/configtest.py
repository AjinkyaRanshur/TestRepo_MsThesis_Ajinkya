import torch

batch_size = 40
epochs = 200
seed = 42
lr = 0.0001
momentum = 0.9
classification_datasetpath="data/visual_illusion_dataset"
recon_datasetpath="/home/ajinkyar/datasets"
training_condition = "illusion_testing"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classification_neurons = 6

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {device}")
    print(f"CUDA device name:{torch.cuda.get_device_name(device) }")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
print(f"Using device: {device}")

timesteps = 100

gammaset = [[0.33, 0.33, 0.33, 0.33]]  # pattern: Uniform
betaset = [[0.33, 0.33, 0.33, 0.33]]  # pattern: Uniform
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pc_recon10_Uniform_seed42_chk15_class_t10_Uniform_seed42"
noise_type="s&p"
noise_param=0.0

experiment_name = "Testing class_pc_recon10_Uniform_seed42_chk15_Uniform_seed42 with Uniform pattern at 10 timesteps"
load_model_path="/home/ajinkyar/ml_models"
save_model_path="/home/ajinkyar/ml_models"



base_recon_model = "pc_recon10_Uniform_seed42"
checkpoint_epoch = 40



