#!/usr/bin/env bash
#SBATCH --qos=img
#SBATCH --partition=imgcomputeq
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=12G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=tsa_job
#SBATCH --output=tsa_job.out
#SBATCH --export=NONE

module load fsl-img
module load conda-img
source activate dwi-tracts

echo "Beginning execution at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Running TSA computation script..."
python -u compute_tract_metrics.py /share/ConnLS/ADNI/dwi-tracts/project/config_tracts_lc.json
