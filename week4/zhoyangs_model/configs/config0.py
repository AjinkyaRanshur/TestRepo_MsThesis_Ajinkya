import torch
batch_size = 40
epochs = 150
seed = 42
lr = 0.00005
momentum = 0.9
<<<<<<< HEAD
datasetpath = '/home/ajinkya/projects/datasets'
=======
datasetpath = '/home/ajinkyar/datasets'
>>>>>>> feac1f8acbfc652eba97239f4f7e66756e8d96b7
training_condition = "pre_training"
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

<<<<<<< HEAD
gammaset=[[0.33, 0.33, 0.33]]
betaset = [[0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01]]

model_name = "pre_training"
=======
gammaset=[[0.33, 0.33, 0.33, 0.33]]
betaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pretraining"
>>>>>>> feac1f8acbfc652eba97239f4f7e66756e8d96b7
noise_type="gauss"
noise_param=0.0
experiment_name = "zhoyang's model test run"

