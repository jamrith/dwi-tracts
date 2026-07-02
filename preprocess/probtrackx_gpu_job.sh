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

# --export=NONE clears the environment including MODULEPATH.
# Source the site lmod init to restore it before any module load calls.
if [ -f /etc/profile.d/z00_lmod.sh ]; then
    source /etc/profile.d/z00_lmod.sh
fi

# Add the imaging-specific modulefile path if missing
if [[ ":$MODULEPATH:" != *":/gpfs01/software/imaging/modulefiles:"* ]]; then
    export MODULEPATH="/gpfs01/software/imaging/modulefiles:${MODULEPATH}"
fi

if ! command -v module &>/dev/null; then
    echo "ERROR: module command unavailable on $(hostname) — aborting so job is marked failed" >&2
    exit 1
fi

if [ -z "$MODULEPATH" ]; then
    echo "ERROR: MODULEPATH is empty on $(hostname) — aborting so job is marked failed" >&2
    exit 1
fi

module load cuda-uoneasy/11.8.0
module load fsl-img/6.0.6.3 # Slightly outdated version
module load conda-img/python3.7

if [ -z "$FSLOUTPUTTYPE" ]; then
    echo "ERROR: FSLOUTPUTTYPE not set after module load — FSL did not load correctly on $(hostname)" >&2
    exit 1
fi

# Use a modified version of $FSLDIR
export FSLDIR="/share/ConnLS/ADNI/fsl_mod"
export PATH="/share/ConnLS/ADNI/fsl_mod/bin":$PATH

cmd="./run_probtrackx.py $1 $2"

echo $cmd
eval $cmd
