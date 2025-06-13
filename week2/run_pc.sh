#!/bin/bash

#SBATCH --job-name=pc_pred_code
#SBATCH --ntasks=1
#SBATCH --error=ar_pred_pc.err
#SBATCH --output=ar_pred_pc.out
#SBATCH --mail-user=ranshur.ajinkya@students.iiserpune.ac.in
#SBATCH --mail-type=END

cd $SLURM_SUBMIT_DIR

conda info

old_ifs=$IFS
IFS='
'
for cmd in `cat cmd_list`
do
srun -N1 -n1 -c1 --exclusive bash -c ${cmd} &
done
wait
IFS=$old_ifs

