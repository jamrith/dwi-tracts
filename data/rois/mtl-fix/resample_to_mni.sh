#!/bin/bash

# Script to resample 193x229x193 files to standard MNI 182x218x182 space
# Uses flirt to handle orientation differences and resampling

REFERENCE="/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz"

# List of files to resample (193x229x193 -> 182x218x182)
FILES=*.nii.gz

echo "Resampling files to MNI152 1mm space (182x218x182)..."
echo "Reference: $REFERENCE"
echo ""

for file in $FILES; do
    if [ ! -f "$file" ]; then
        echo "WARNING: $file not found, skipping..."
        continue
    fi

    # Create output filename
    basename="${file%.nii.gz}"
    output="${basename}_mni.nii.gz"

    echo "Processing: $file -> $output"

    flirt -in "$file" \
          -ref "$REFERENCE" \
          -out "$output" \
          -applyxfm \
          -usesqform \
          -interp nearestneighbour

    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
    fi
    echo ""
done

echo "Resampling complete!"
echo ""
echo "Verifying output dimensions..."
for file in "${FILES[@]}"; do
    basename="${file%.nii.gz}"
    output="${basename}_mni.nii.gz"
    if [ -f "$output" ]; then
        echo "=== $output ==="
        fslinfo "$output" | grep -E "^dim[1-3]|^pixdim[1-3]"
    fi
done
