import torch
batch_size = 40
epochs = 10
seed = 42
lr = 0.00005
momentum = 0.9
datasetpath = '/home/ajinkya/datasets'
training_condition = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {device}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

timesteps = 10

gammaset = [[0.53, 0.53, 0.53, 0.53]]
betaset = [[0.13, 0.13, 0.13, 0.13]]
alphaset = [[0.09, 0.09, 0.09, 0.09]]

model_name = "pc_class_train_balanced"
noise_type = "s&p"
noise_param = 0.00
experiment_name = "Testing Model on Beta 0.13 and Gamma 0.53 and Alpha 0.09"

load_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/classification_models"
save_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/classification_models"

