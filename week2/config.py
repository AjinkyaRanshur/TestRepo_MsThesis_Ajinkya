import torch
batch_size=4
epochs=10
seed=1
lr=0.001
momentum=0.9
#training_condition="ff_fb_train"
training_condition="pc_train"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps=10
load_model=False
save_model=True
#hyp_dict={'Gamma= 0.3,0.3,0.3\n Beta=0.3,0.3,0.3\n alpha=0.01,0.01,0.01\n ':
#        [[0.3,0.3,0.3],[0.3,0.3,0.3],[0.01,0.01,0.01]}

hyp_dict={'Gamma= 0.4,0.2,0.8\n Beta=0.5,0.3,0.2\n alpha=0.01,0.01,0.01\n ':
        [[0.4,0.2,0.8],[0.5,0.3,0.2],[0.01,0.01,0.01]],
        'Gamma= 0.2,0.2,0.2\n Beta=0.2,0.4,0.5\n alpha=0.01,0.01,0.01\n ':
        [[0.2,0.2,0.2],[0.2,0.4,0.5],[0.01,0.01,0.01]],
        'Gamma= 0.5,0.5,0.5\n Beta=0.5,0.5,0.5\n alpha=0.01,0.01,0.01\n ':
        [[0.5,0.5,0.5],[0.5,0.5,0.5],[0.01,0.01,0.01]]}





