import numpy as np

iters = 0

insert_gamma=[0.13, 0.43, 0.13, 0.33]
insert_beta=[0.43, 0.13, 0.43, 0.33]

for spnoise in np.arange(0, 1.0, 0.2):
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
noise_param={spnoise}
experiment_name = "pc_model_pyramid_scheme_part_d_sp_{spnoise}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1

for gnoise in np.arange(0, 0.08, 0.02):
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
noise_param={gnoise}
experiment_name = "pc_model_pyramid_scheme_part_d_gauss_{gnoise}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1


for spnoise in np.arange(0, 1.0, 0.2):
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
noise_param={spnoise}
experiment_name = "ff_fb_model_pyramid_scheme_part_d_sp_{spnoise}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1

for gnoise in np.arange(0, 0.08, 0.02):
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
noise_param={gnoise}
experiment_name = "ff_fb_model_pyramid_scheme_part_d_gauss_{gnoise}"

'''

    with open(f"config{iters}.py", "w") as f:
        f.write(config_code)

    iters += 1

print("Number of files Generated",iters)
