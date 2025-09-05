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

gammaset = [[0.23, 0.23, 0.23, 0.23]]
betaset = [[0.43, 0.43, 0.43, 0.43]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pc_class_train_balanced"
noise_type = "s&p"
noise_param = 0.00
experiment_name = "Testing Model on Beta 0.43 and Gamma 0.23 and Alpha 0.01"

load_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/classification_models"
save_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/classification_models"

