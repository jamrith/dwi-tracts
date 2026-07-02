#!/usr/bin/env bash
#SBATCH --qos=img
#SBATCH --partition=imgcomputeq
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=glm_limbic
#SBATCH --output=glm_limbic.out
#SBATCH --export=NONE

module load fsl-img
module load conda-img
source activate dwi-tracts

echo "Beginning execution at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Running GLM for LC-limbic network..."
python -u run_glm_limbic.py

echo "Done at $(date)"
