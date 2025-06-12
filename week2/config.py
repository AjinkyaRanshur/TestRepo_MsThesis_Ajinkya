import torch
batch_size=4
epochs=5
seed=30
lr=0.001
momentum=0.9
#training_condition="ff_fb_train"
training_condition="pc_train"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps=4



