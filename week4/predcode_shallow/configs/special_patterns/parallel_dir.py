import numpy as np

iters = 0

insert_gamma=[0.33, 0.33, 0.33, 0.33]
insert_beta=[0.33, 0.33, 0.33, 0.33]

insert_scheme="control_models"


# salt and pepper in pc model
for spnoise in np.arange(0, 0.08, 0.02):
    config_code = f'''import torch
batch_size = 128
epochs = 70
seed = 42
lr = 0.001
momentum = 0.9
datasetpath = '/home/ajinkyar/datasets'
training_condition = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {{torch.cuda.is_available()}}")
print(f"CUDA device count: {{torch.cuda.device_count()}}")
if torch.cuda.is_available():
    print(f"CUDA device name: {{torch.cuda.get_device_name(0)}}")
    print(f"CUDA memory allocated: {{torch.cuda.memory_allocated(0) / 1024**3:.2f}} GB")
print(f"Using device: {{device}}")

timesteps = 10
load_model = False
save_model = True

gammaset=[{insert_gamma}]
betaset = [{insert_beta}]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "pc_model"
noise_type="s&p"
noise_param={spnoise:.3f}
experiment_name = "control_pc_model_sp_{spnoise:.3f}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1

# gauss in pc model
for gnoise in np.arange(0,1.0, 0.2):
    config_code = f'''import torch
batch_size = 128
epochs = 70
seed = 42
lr = 0.001
momentum = 0.9
datasetpath = '/home/ajinkyar/datasets'
training_condition = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {{torch.cuda.is_available()}}")
print(f"CUDA device count: {{torch.cuda.device_count()}}")
if torch.cuda.is_available():
    print(f"CUDA device name: {{torch.cuda.get_device_name(0)}}")
    print(f"CUDA memory allocated: {{torch.cuda.memory_allocated(0) / 1024**3:.2f}} GB")
print(f"Using device: {{device}}")

timesteps = 10
load_model = False
save_model = True

gammaset=[{insert_gamma}]
betaset = [{insert_beta}]
alphaset = [[0.01, 0.01, 0.01, 0.01]]


model_name = "pc_model"
noise_type="gauss"
noise_param={gnoise:.3f}
experiment_name = "control_pc_model_gauss_{gnoise:.3f}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1

#Salt and Pepper in ff and fb train

for spnoise in np.arange(0, 0.08, 0.02):
    config_code = f'''import torch
batch_size = 128
epochs = 70
seed = 42
lr = 0.001
momentum = 0.9
datasetpath = '/home/ajinkyar/datasets'
training_condition = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {{torch.cuda.is_available()}}")
print(f"CUDA device count: {{torch.cuda.device_count()}}")
if torch.cuda.is_available():
    print(f"CUDA device name: {{torch.cuda.get_device_name(0)}}")
    print(f"CUDA memory allocated: {{torch.cuda.memory_allocated(0) / 1024**3:.2f}} GB")
print(f"Using device: {{device}}")

timesteps = 10
load_model = False
save_model = True

gammaset=[{insert_gamma}]
betaset = [{insert_beta}]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "ff_fb_model"
noise_type="s&p"
noise_param={spnoise:.3f}
experiment_name = "control_ff_fb_model_sp_{spnoise:.3f}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1
    

#Gauss in ff and fb train

for gnoise in np.arange(0,1.0, 0.2):
    config_code = f'''import torch
batch_size = 128
epochs = 70
seed = 42
lr = 0.001
momentum = 0.9
datasetpath = '/home/ajinkyar/datasets'
training_condition = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {{torch.cuda.is_available()}}")
print(f"CUDA device count: {{torch.cuda.device_count()}}")
if torch.cuda.is_available():
    print(f"CUDA device name: {{torch.cuda.get_device_name(0)}}")
    print(f"CUDA memory allocated: {{torch.cuda.memory_allocated(0) / 1024**3:.2f}} GB")
print(f"Using device: {{device}}")

timesteps = 10
load_model = False
save_model = True

gammaset=[{insert_gamma}]
betaset = [{insert_beta}]
alphaset = [[0.01, 0.01, 0.01, 0.01]]


model_name = "ff_fb_model"
noise_type="gauss"
noise_param={gnoise:.3f}
experiment_name = "control_ff_fb_model_gauss_{gnoise:.3f}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1
        
print("Number of files Generated",iters)
