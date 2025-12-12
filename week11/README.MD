# AGENDA TO TAKE CARE OF

## Overview
We have fu*** around a lot so now you get the basic idea about what to do some scripts require new features to be added some new script need to be added to make the general experience smoother here a list of things we need to do


- make main.py shorter and also fix dataloading procedures
- if you have time make network.py in a way that it can use any architecture just by loading a json file
- need ill_pc_train.py to handle the new dataset
- eval_and_plotting.py(needs fixes recon loss is for cifar10 not illusion dataset)
- add_noise.py unchanged
- for loading the new dataset create_custom.py
- configs (needs polishing)
- gui-interface.py this is ridiculously long make it modular split it into the functions marked with an arrow
- batch_submission.py this will be helpful for slurm stuff
-> grid search either layer by layer change or all the layers changed
-> pattern formation unchanged
-> reconstruction comparison given an input make it again I forgot how to do it 
-> model tracking 
-> batch file creation for creating slurm submission files
-> monitoring stuff if I'm making interactive jobs about what is happening don't know what I'm doing with this
-> hyperparameters optimizable this is pretty cool gonna surprise andrea in the end with this,the core idea is we will train models with different levels of noise and seeds and then keep the hyperparameters optimizable starting from randomly intialized values and then we will use illusions index to see if the models above an illusion index support our hypothesis or not 



## Future Directions




