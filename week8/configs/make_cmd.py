import numpy as np

iters = 0

# use linspace to avoid floating point drift
gamma_values = np.round(np.linspace(0.13, 0.63, 6), 2)
beta_values = np.round(np.linspace(0.13, 0.63, 6), 2)
alpha_values = np.round(np.linspace(0.01, 0.3, 30), 3)

for gamma in gamma_values:
    gamma=float(gamma)
    insert_gamma = [gamma] * 4
    for beta in beta_values:
        beta=float(beta)
        insert_beta = [beta] * 4
        for alpha in alpha_values:
            alpha=float(alpha)
            insert_alpha = [alpha] * 4
            config_code = f'''import torch
batch_size = 40
epochs = 10
seed = 42
lr = 0.00005
momentum = 0.9
datasetpath = '/home/ajinkya/datasets'
training_condition = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {{torch.cuda.is_available()}}")
print(f"CUDA device count: {{torch.cuda.device_count()}}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {{device}}")
    print(f"CUDA device name: {{torch.cuda.get_device_name(0)}}")

timesteps = 10

gammaset = [{insert_gamma}]
betaset = [{insert_beta}]
alphaset = [{insert_alpha}]

model_name = "pc_class_train_balanced_t1"
noise_type = "s&p"
noise_param = 0.00
experiment_name = "Testing Model on Beta {beta} and Gamma {gamma} and Alpha {alpha}"

load_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/classification_models/recon_t1_timesteps1"
save_model_path="/home/ajinkya/projects/TestRepo_MsThesis_Ajinkya/week8/models/classification_models/recon_t1_timesteps1"
illusion_dataset_bool=False

'''
            with open(f"config{iters}.py", "w") as f:
                f.write(config_code)
            iters += 1

print("Number of files Generated", iters)

