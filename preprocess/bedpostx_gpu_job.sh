#!/usr/bin/bash

#SBATCH --partition=imgcomputeq
# This specifies type of node job will use

#SBATCH --nodes=1
# This specifies job uses 1 node

#SBATCH --ntasks-per-node=1
# This specifies job only use 1 core on the node

#SBATCH --mem=2g
# This specifies maximum memory use will be 2 gigabytes

#SBATCH --time=00:10:00
# This specifies job will last no longer than 10 minutes

#SBATCH -o logs/bedpostx_gpu-%j.out

#SBATCH --qos=img

#SBATCH --export=NONE

# Load relevant modules here
module load fsl-img/6.0.6.3
module load conda-img/python3.7
source activate dwi-tracts

# Preprocessing steps
cmd="python run_bedpostx_gpu.py $1 $2"

echo $cmd
eval $cmd
