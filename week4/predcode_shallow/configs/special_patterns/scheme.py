import numpy as np

iters = 0

for gamma in np.arange(0.13, 0.63, 0.1):
    gamma=float(gamma)
    gamma=round(gamma,2)
    insert_gamma = [gamma, gamma, gamma, gamma]
    for beta in np.arange(0.13, 0.63, 0.1):
        beta=float(beta)
        beta=round(beta,2)
        insert_beta = [beta, beta, beta, beta]
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
    print(
    f"CUDA memory allocated: {{torch.cuda.memory_allocated(0) / 1024**3:.2f}} GB")
print(f"Using device: {{device}}")

timesteps = 10
load_model = False
save_model = True

gammaset = [{insert_gamma}]
betaset = [{insert_beta}]
alphaset = [[0.01, 0.01, 0.01, 0.01]]

model_name = "ff_fb_model"
noise_type = "s&p"
noise_param = 0.00
experiment_name = "fffb_model_gamma_{gamma}_beta_{beta}"
'''
        with open(f"config{iters}.py", "w") as f:
            f.write(config_code)
        iters += 1

print("Number of files Generated", iters)

