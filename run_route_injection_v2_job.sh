#!/usr/bin/env bash
# Wrapper for project/run_route_injection_v2.py (medial/lateral, combined LR).
if [ -f /etc/profile.d/z00_lmod.sh ]; then
    source /etc/profile.d/z00_lmod.sh
fi
if [[ ":$MODULEPATH:" != *":/gpfs01/software/imaging/modulefiles:"* ]]; then
    export MODULEPATH="/gpfs01/software/imaging/modulefiles:${MODULEPATH}"
fi

module load fsl-img
module load conda-img
source activate dwi-tracts

cd /gpfs01/imgshare/ConnLS/ADNI/dwi-tracts
python3 -u /gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project/run_route_injection_v2.py "$@"
