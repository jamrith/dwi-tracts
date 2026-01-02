#!/bin/bash

# Script to resample a mask from 0.5mm^3 to 1mm^3 resolution and binarize at 0.5

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_mask.nii.gz> <output_mask.nii.gz>"
    exit 1
fi

input=$1
output=$2

# Resample from 0.5mm to 1mm and binarize at 0.5
flirt -in ${input} -ref ${input} -out ${output} -applyisoxfm 1 -interp trilinear
fslmaths ${output} -thr 0.01 -bin ${output}

