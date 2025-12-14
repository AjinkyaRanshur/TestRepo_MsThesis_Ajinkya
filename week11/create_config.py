import os
import sys

BASE_DIR="configs/base_config.py"
CONFIG_FILE=BASE_DIR

def update_config(
    gamma_pattern,
    beta_pattern,
    pattern_name,
    model_name,
    timesteps,
    train_cond,
    last_neurons,
    seed,
    lr
):
    """Update the config file with new parameters."""
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()

    with open(CONFIG_FILE, "w") as f:
        for line in lines:
            stripped = line.strip()

            if stripped.startswith("gammaset"):
                f.write(
                    f"gammaset = [[{', '.join(f'{g:.2f}' for g in gamma_pattern)}]]  "
                    f"# pattern: {pattern_name}\n"
                )

            elif stripped.startswith("betaset"):
                f.write(
                    f"betaset = [[{', '.join(f'{b:.2f}' for b in beta_pattern)}]]  "
                    f"# pattern: {pattern_name}\n"
                )

            elif stripped.startswith("model_name"):
                f.write(f'model_name = "{model_name}"\n')
	    
            elif stripped.startswith("seed"):
                f.write(f'seed = {seed}\n')

	    elif stripped.startswith("lr"):
                f.write(f'lr = {lr}\n')

            elif stripped.startswith("timesteps"):
                f.write(f"timesteps = {timesteps}\n")
	    
            elif stripped.startswith("classification_neurons"):
                f.write(f"classification_neurons = {last_neurons}\n")

            elif stripped.startswith("experiment_name"):
                f.write(
                    f'experiment_name = "Testing {model_name} with {pattern_name} '
                    f'pattern at {timesteps} timesteps"\n'
                )

            elif stripped.startswith("training_condition"):
              if train_cond == None:
                f.write(f'training_condition = {train_cond}\n')
              else:
                f.write(f'training_condition = "{train_cond}"\n')


            else:
                f.write(line)

#USED TO CREATE MULTIPLE CONFIG FILES FROM JSON FILE IN CASE WE NEED TO PERFORM MULTIPLE EXPERIMENTS
def create_config_files(seeds,patterns,train_cond,epochs,lr_list,timesteps,model_name,last_neurons):

	config_paths=[]
	
	exp_id=0

	# Pattern definitions
	PATTERNS = {
    	"Uniform": {
        	"gamma": [0.33, 0.33, 0.33, 0.33],
        	"beta": [0.33, 0.33, 0.33, 0.33]
    	},
    	"Gamma Increasing": {
        	"gamma": [0.13, 0.33, 0.53, 0.33],
        	"beta": [0.33, 0.33, 0.33, 0.33]
    	},
    	"Gamma Decreasing": {
        	"gamma": [0.53, 0.33, 0.13, 0.33],
        	"beta": [0.33, 0.33, 0.33, 0.33]
    	},
    	"Beta Increasing": {
        	"gamma": [0.33, 0.33, 0.33, 0.33],
        	"beta": [0.13, 0.33, 0.53, 0.33]
    	},
    	"Beta Decreasing": {
        	"gamma": [0.33, 0.33, 0.33, 0.33],
        	"beta": [0.53, 0.33, 0.13, 0.33]
    	},
    	"Beta Inc & Gamma Dec": {
        	"gamma": [0.53, 0.33, 0.13, 0.33],
        	"beta": [0.13, 0.33, 0.53, 0.33]
    	}
	}
	
	for seed,pattern,lr,timestep,model in itertools.product(seeds,patterns,lr_list,timesteps,model_name):
		model_name=generate_model_name(pattern,seed,train_cond,recon_timesteps)
		json_file=model_tracker()	
		gamma_pattern=PATTERNS[pattern]["gamma"]
		beta_pattern=PATTERNS[pattern]["beta"]
		
		cfg_path=f"configs/generated/config_{exp_id}.py"
		
		CONFIG_FILE=cfg_path
		with open(BASE_DIR) as f:
			base=f.read()
		with open(cfg_path,"w") as f:
			f.write(base)

		update_config(gamma_pattern,beta_pattern,model_name,timesteps,train_cond,seed,lr)
		
		config_paths.append()
		exp_id += 1

	print(f"Generated {len(config_paths)} config files")

return config_paths











