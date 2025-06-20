#!/bin/bash

#SBATCH --job-name=pred_code
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --error=ar_pred.err
#SBATCH --output=ar_pred.out
#SBATCH --mail-user=ranshur.ajinkya@students.iiserpune.ac.in
#SBATCH --mail-type=END

cd $SLURM_SUBMIT_DIR

conda info

srun python3 main.py --config config1
srun echo "Job excuted for $(hostname)"

srun python3 main.py --config config2
srun echo "Job excuted for $(hostname)"

srun python3 main.py --config config3
srun echo "Job excuted for $(hostname)"

srun python3 main.py --config config4
srun echo "Job excuted for $(hostname)"

srun python3 main.py --config config5
srun echo "Job excuted for $(hostname)"
