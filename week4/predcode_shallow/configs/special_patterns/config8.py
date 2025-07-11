import torch
batch_size = 128
epochs = 70
seed = 42
lr = 0.001
momentum = 0.9
datasetpath = '/home/ajinkyar/datasets'
training_condition = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Using device: {device}")

timesteps = 10
load_model = False
save_model = True

gammaset=[[0.13, 0.43, 0.13, 0.33]]
betaset = [[0.43, 0.13, 0.43, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]


model_name = "pc_model"
noise_type="gauss"
noise_param=0.800
experiment_name = "pc_model_Parallel_part_d_gauss_0.800"

