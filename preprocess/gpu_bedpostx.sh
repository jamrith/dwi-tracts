#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH --job-name=bedpostx
#SBATCH --partition=imgampereq,imgvoltaq,imgpascalq
#SBATCH --qos=img
#SBATCH --account=uon-imaging
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=30g
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# Usage: sbatch gpu_bedpostx.sh 067_S_2304_2019-08-09

echo "running on $(hostname)"
cd "/share/ConnLS/ADNI/derived/"

module load extension/imaging
module load fsl-img/6.0.7.x

/share/ConnLS/ADNI/dwi-tracts/preprocess/bedpostx_gpu $1/dwi -QSYS 2
rc=$?

# bedpostx_gpu itself doesn't write this project's completion marker --
# run_bedpostx_postproc.py requires it before it will proceed, same
# convention as preproc.done.
if [ $rc -eq 0 ] && [ -d "$1/dwi.bedpostX" ]; then
  mkdir -p "$1/dwi/flags"
  touch "$1/dwi/flags/bedpostx.done"
  echo "Wrote flag file $1/dwi/flags/bedpostx.done"
else
  echo "bedpostx_gpu did not complete successfully for $1 (rc=$rc) -- not writing flag"
fi

