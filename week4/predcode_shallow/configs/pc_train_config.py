import torch
batch_size = 128
epochs = 70
seed = 42
lr = 0.001
momentum = 0.9
datasetpath = '/home/ajinkyar/datasets'
training_condition = "pc_train"
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

gammaset=[[0.33, 0.33, 0.33, 0.33]]
betaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

experiment_name = "PredictiveCoding_Train"
model_name = "pc_model"
noise_type="s&p"
noise_param=0

