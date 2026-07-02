#!/usr/bin/env bash
#SBATCH --qos=img
#SBATCH --partition=imgcomputeq
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=tsa_hypothal
#SBATCH --output=tsa_limbic_hypothal.out
#SBATCH --export=NONE

module load fsl-img
module load conda-img
source activate dwi-tracts

echo "Beginning execution at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"

CONFIG="project/config_tracts_lc_limbic_hypothal.json"

echo "Running hypothal-only tract metrics (polylines) for LC-limbic network..."
python -u compute_tract_metrics.py "$CONFIG"

echo "Done at $(date)"
