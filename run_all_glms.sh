#!/usr/bin/env bash
#SBATCH --qos=img
#SBATCH --partition=imgcomputeq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=tsa_glm
#SBATCH --output=glms.out
#SBATCH --export=NONE

module load fsl-img
module load conda-img
source activate dwi-tracts

echo "Beginning execution at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Running all GLMs..."
python -u run_all_glms.py
