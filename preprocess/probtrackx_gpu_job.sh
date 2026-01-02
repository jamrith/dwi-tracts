#!/usr/bin/bash

#SBATCH --qos=img
#SBATCH --partition=imgvoltaq
# This specifies type of node job will use

#SBATCH --nodes=1
# This specifies job uses 1 node

#SBATCH --ntasks-per-node=2
# This specifies job only use 2 cores on the node

#SBATCH --mem=20g
# This specifies maximum memory use will be 20 gigabytes

#SBATCH --time=30:00:00
# This specifies job will last no longer than 30 hours

#SBATCH --gres=gpu:1

#SBATCH --export=NONE

#SBATCH -o logs/probtrackx-%j.out
#SBATCH -e logs/probtrackx-%j.err


module load cuda-uoneasy/11.8.0
module load fsl-img/6.0.6.3 # Slightly outdated version
module load conda-img/python3.7

# Use a modified version of $FSLDIR
export FSLDIR="/share/ConnLS/ADNI/fsl_mod"
export PATH="/share/ConnLS/ADNI/fsl_mod/bin":$PATH

cmd="./run_probtrackx.py $1 $2"

echo $cmd
eval $cmd
