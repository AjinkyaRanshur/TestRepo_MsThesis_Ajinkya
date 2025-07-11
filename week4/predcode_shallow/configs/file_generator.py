import numpy as np

iters = 0 
md_name="ff_fb_model"

##For Gamma Modulation with noisy images

for layer_iter in range(4):
    for spnoise in np.arange(0, 0.08, 0.02):
        for gamma in np.arange(0.13, 0.63, 0.1):
            insert_val = [0.33, 0.33, 0.33, 0.33]
            insert_val[layer_iter] = float(round(gamma, 2))
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

gammaset=[{insert_val}]
betaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "{md_name}"
noise_type="s&p"
noise_param={spnoise}
experiment_name = "Gamma_layer_{layer_iter}_perturbation_{gamma:.2f}_sp_{spnoise:.2f}"
'''

            with open(f"config{iters}.py", "w") as f:
                f.write(config_code)
            iters += 1

    for gnoise in np.arange(0, 1.0, 0.2):
        for gamma in np.arange(0.13, 0.63, 0.1):
            insert_val = [0.33, 0.33, 0.33, 0.33]
            insert_val[layer_iter] = float(round(gamma, 2))
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

gammaset=[{insert_val}]
betaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]


model_name = "{md_name}"
noise_type="gauss"
noise_param={gnoise}
experiment_name = "Gamma_layer_{layer_iter}_perturbation_{gamma:.2f}_gauss_{gnoise:.2f}"
'''

            with open(f"config{iters}.py", "w") as f:
                f.write(config_code)
            iters += 1

##For beta modulation with noisy images

for layer_iter in range(3):
    for spnoise in np.arange(0, 0.08, 0.02):
        for beta in np.arange(0.13, 0.63, 0.1):
            insert_val = [0.33, 0.33, 0.33, 0.33]
            insert_val[layer_iter] = float(round(beta, 2))
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

betaset=[{insert_val}]
gammaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "{md_name}"
noise_type="s&p"
noise_param={spnoise}
experiment_name = "Beta_layer{layer_iter}_perturbation_{beta:.2f}_sp_{spnoise:.2f}"
'''

            with open(f"config{iters}.py", "w") as f:
                f.write(config_code)
            iters += 1

    for gnoise in np.arange(0, 1.0, 0.2):
        for beta in np.arange(0.13, 0.63, 0.1):
            insert_val = [0.33, 0.33, 0.33, 0.33]
            insert_val[layer_iter] = float(round(beta, 2))
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

betaset=[{insert_val}]
gammaset = [[0.33, 0.33, 0.33, 0.33]]
alphaset = [[0.01, 0.01, 0.01, 0.01]]


model_name = "{md_name}"
noise_type="gauss"
noise_param={gnoise}
experiment_name = "Beta_layer{layer_iter}_perturbation_{gamma:.2f}_gauss_{gnoise:.2f}"
'''

            with open(f"config{iters}.py", "w") as f:
                f.write(config_code)
            iters += 1            

     
print("Number of files Generated",iters)
