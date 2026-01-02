#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH --job-name=bedpostx
#SBATCH --partition=imgvoltaq
#SBATCH --qos=img
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

