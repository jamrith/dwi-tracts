#!/usr/bin/bash

#SBATCH --qos=img
#SBATCH --partition=imgvoltaq,imgpascalq
# This specifies type of node job will use

#SBATCH --nodes=1
# This specifies job uses 1 node

#SBATCH --ntasks-per-node=1
# This specifies job only use 2 cores on the node

#SBATCH --mem=8g
# This specifies maximum memory use will be 20 gigabytes

#SBATCH --time=05:00:00
# This specifies job will last no longer than 30 hours

#SBATCH --gres=gpu:1

#SBATCH --export=NONE

#SBATCH -o logs/probtrackx-%j.out

# Load relevant modules here
module load fsl-img/6.0.7.x
module load conda-img/python3.7
module load cuda-uoneasy/11.8.0

# Use a modified version of $FSLDIR
export FSLDIR="/share/ConnLS/ADNI/fsl_mod"
export PATH="/share/ConnLS/ADNI/fsl_mod/bin":$PATH

cmd="./run_probtrackx.py $1 $2"

echo $cmd
eval $cmd
