import torch
batch_size=4
epochs=10
seed=1
lr=0.001
momentum=0.9
#training_condition="ff_fb_train"
training_condition="pc_train"

# Fixed device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Using device: {device}")


timesteps=10
load_model=False
save_model=True

# hyp_dict={'Gamma= 0.3,0.3,0.3\n Beta=0.3,0.3,0.3\n alpha=0.01,0.01,0.01\n ':
        #   [[0.3,0.3,0.3],[0.3,0.3,0.3],[0.01,0.01,0.01]]}

# gammaset=[[0.3,0.3,0.3]]

# betaset=[[0.3,0.3,0.3]]

# alphaset=[[0.01,0.01,0.01]]

gammaset=[[0.4,0.2,0.8],[0.2,0.2,0.2],[0.5,0.5,0.5]]

betaset=[[0.5,0.3,0.2],[0.2,0.4,0.5],[0.5,0.5,0.5]]

alphaset=[[0.01,0.01,0.01],[0.01,0.01,0.01],[0.01,0.01,0.01]]

# hyp_dict={'Gamma= 0.4,0.2,0.8\n Beta=0.5,0.3,0.2\n alpha=0.01,0.01,0.01\n ':
#         [[0.4,0.2,0.8],[0.5,0.3,0.2],[0.01,0.01,0.01]],
#         'Gamma= 0.2,0.2,0.2\n Beta=0.2,0.4,0.5\n alpha=0.01,0.01,0.01\n ':
#         [[0.2,0.2,0.2],[0.2,0.4,0.5],[0.01,0.01,0.01]],
#         'Gamma= 0.5,0.5,0.5\n Beta=0.5,0.5,0.5\n alpha=0.01,0.01,0.01\n ':
#         [[0.5,0.5,0.5],[0.5,0.5,0.5],[0.01,0.01,0.01]]}





