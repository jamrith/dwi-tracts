#!/bin/bash
# Prototype: warp one subject's ASHS BA35 (transentorhinal) label into Mean3G
# template space, reusing existing transforms only.
#
# Chain (all transforms already exist per subject):
#   ASHS seg (T2/tse space)
#     --(inverse of mind/s0_to_t2.mat)-->  DWI space   [flirt applyxfm, NN]
#     --(reg3G/FA_warp2Mean3G.nii.gz)   -->  Mean3G     [applywarp, NN]
#
# Usage: prototype_ashs_to_mean3g.sh <subject> <side> <label> <roi_name>
set -euo pipefail

FSL=/software/imaging/fsl/6.0.6.3
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSL/bin:$PATH"

ROOT=/gpfs01/imgshare/ConnLS/ADNI
SUBJECT="${1:-002_S_0413_2019-08-27}"
SIDE="${2:-left}"          # left | right
LABEL="${3:-11}"           # 11 = BA35 (transentorhinal)
NAME="${4:-BA35_L}"        # output ROI name

SD="$ROOT/derived/$SUBJECT"
SEG="$SD/ashs/final/ashs_${SIDE}_lfseg_corr_usegray.nii.gz"
S0_TO_T2="$SD/mind/s0_to_t2.mat"
FA_REF="$SD/dwi/dti_FA.nii.gz"
FA_WARP="$SD/dwi/reg3G/FA_warp2Mean3G.nii.gz"
MEAN3G="$ROOT/dwi-tracts/utils/Mean3G.nii.gz"

OUT="$ROOT/dwi-tracts/tmp/native_roi_proto/$SUBJECT"
mkdir -p "$OUT"

echo "== Subject: $SUBJECT | side=$SIDE label=$LABEL name=$NAME =="
for f in "$SEG" "$S0_TO_T2" "$FA_REF" "$FA_WARP" "$MEAN3G"; do
    [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

# 0) Binarise the requested label in ASHS/T2 space
echo "[0] extract label $LABEL from ASHS seg"
fslmaths "$SEG" -thr "$LABEL" -uthr "$LABEL" -bin "$OUT/${NAME}_t2.nii.gz"
nvox=$(fslstats "$OUT/${NAME}_t2.nii.gz" -V | awk '{print $1}')
echo "    voxels in T2 space: $nvox"
[ "$nvox" -gt 0 ] || { echo "ERROR: empty label" >&2; exit 1; }

# 1) T2 -> DWI  (invert the S0->T2 flirt matrix, apply NN)
echo "[1] invert s0_to_t2.mat -> t2_to_s0.mat"
convert_xfm -omat "$OUT/t2_to_s0.mat" -inverse "$S0_TO_T2"
echo "    T2 seg -> DWI space (nearest-neighbour)"
flirt -in "$OUT/${NAME}_t2.nii.gz" -ref "$FA_REF" -applyxfm -init "$OUT/t2_to_s0.mat" \
      -interp nearestneighbour -out "$OUT/${NAME}_dwi.nii.gz"
fslmaths "$OUT/${NAME}_dwi.nii.gz" -bin "$OUT/${NAME}_dwi.nii.gz"

# 2) DWI -> Mean3G  (apply reg3G warp, NN), resample onto exact Mean3G grid
echo "[2] DWI seg -> Mean3G (applywarp, nearest-neighbour)"
applywarp --in="$OUT/${NAME}_dwi.nii.gz" --ref="$MEAN3G" --warp="$FA_WARP" \
          --interp=nn --out="$OUT/${NAME}.nii.gz"
fslmaths "$OUT/${NAME}.nii.gz" -bin "$OUT/${NAME}.nii.gz"

# QC: copy template + existing MNI Ent_L for overlay comparison
cp "$MEAN3G" "$OUT/Mean3G.nii.gz"
cp "$ROOT/dwi-tracts/data/rois/mtl-fix/Ent_L.nii.gz" "$OUT/Ent_L_oldMNI.nii.gz" 2>/dev/null || true

echo "== Voxel counts =="
printf "  %-22s %s\n" "T2 space:"    "$(fslstats "$OUT/${NAME}_t2.nii.gz"  -V | awk '{print $1}')"
printf "  %-22s %s\n" "DWI space:"   "$(fslstats "$OUT/${NAME}_dwi.nii.gz" -V | awk '{print $1}')"
printf "  %-22s %s\n" "Mean3G space:" "$(fslstats "$OUT/${NAME}.nii.gz"    -V | awk '{print $1}')"
echo "== Mean3G grid check (must match ROI grid used by dwitracts) =="
fslinfo "$OUT/${NAME}.nii.gz" | grep -E "^dim[1-3]|pixdim[1-3]"
echo "OUTPUT DIR: $OUT"
ls -la "$OUT"
