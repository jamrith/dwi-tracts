#!/usr/bin/env bash
#SBATCH --qos=img
#SBATCH --partition=imgcomputeq
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH --job-name=tsa_single
#SBATCH --output=tsa_single.out
#SBATCH --export=NONE

module load fsl-img
module load conda-img
source activate dwi-tracts

echo "Beginning execution at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"

SUBJECT="941_S_7074_2022-05-03"
CONFIG="/share/ConnLS/ADNI/dwi-tracts/project/config_tracts_lc.json"

echo "Running single-subject TSA for: ${SUBJECT}"
python -u run_single_subject_tsa.py "${CONFIG}" "${SUBJECT}"
