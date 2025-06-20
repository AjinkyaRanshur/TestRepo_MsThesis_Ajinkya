#!/bin/bash

#SBATCH --job-name=pred_code
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2   
#SBATCH --error=ar_pred.err
#SBATCH --output=ar_pred.out
#SBATCH --mail-user=ranshur.ajinkya@students.iiserpune.ac.in
#SBATCH --mail-type=END

cd $SLURM_SUBMIT_DIR

conda info

old_ifs=$IFS
IFS='
'
for cmd in `cat cmd_list`
do
srun bash -c "${cmd}" &
done
wait
IFS=$old_ifs

