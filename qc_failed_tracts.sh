#!/usr/bin/env bash
# qc_failed_tracts.sh
#
# Visual QC for subjects with failed tracts (tsa_failed_tracts.csv).
# Assumed cause: coregistration failure (FA → MNI / Mean3G space).
#
# Run this script on your LOCAL machine, e.g.:
#   bash qc_failed_tracts.sh lpxjr4@hpc.hostname.ac.uk
#
# For each unique failed subject it will:
#   1. scp FA_nlin2Mean3G.nii.gz (FA registered to MNI space)
#   2. scp Mean3G.nii.gz (the MNI reference template)   — once only
#   3. scp the relevant failed-tract ROI masks            — once only
#   4. Open ITK-SNAP:  main = Mean3G,  overlay = FA_nlin2Mean3G,
#                      additional segmentation overlays for each failed ROI
#   5. Wait for ITK-SNAP to close before moving to the next subject
#
# Positional arguments:
#   $1  HPC user@host   (required)
#   $2  Path to tsa_failed_tracts.csv on the HPC
#         (default: /gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/tsa_failed_tracts.csv)
#
# Dependencies (local): ssh, scp, itksnap (or ITK-SNAP)
# Compatible with bash 3+ and zsh.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these if paths differ
# ---------------------------------------------------------------------------
HPC_USER_HOST="${1:-}"
CSV_REMOTE="${2:-/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/tsa_failed_tracts.csv}"

HPC_DERIVED="/share/ConnLS/ADNI/derived"
HPC_TEMPLATE="/share/ConnLS/ADNI/dwi-tracts/utils/Mean3G.nii.gz"
HPC_ROIS_DIR="/share/ConnLS/ADNI/dwi-tracts/data/rois/mtl-fix"

LOCAL_TMP="/tmp/qc_failed_tracts"

ITKSNAP_BIN="/Applications/ITK-SNAP.app/Contents/bin/itksnap"   # override with ITKSNAP_BIN=/path/to/itksnap if needed

# ---------------------------------------------------------------------------
# Argument check
# ---------------------------------------------------------------------------
if [[ -z "$HPC_USER_HOST" ]]; then
    echo "Usage: $0 user@hpc-host [/path/to/tsa_failed_tracts.csv]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Ctrl+C handler — kill any running ITK-SNAP and scp, then exit cleanly
# ---------------------------------------------------------------------------
SNAP_PID=""
cleanup() {
    echo ""
    echo "Interrupted."
    [[ -n "$SNAP_PID" ]] && kill "$SNAP_PID" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Helper: scp a remote file, skip if already present locally
# ---------------------------------------------------------------------------
fetch() {
    local remote="$1"
    local dest="$2"
    if [[ ! -f "$dest" ]]; then
        echo "  Fetching $(basename "$remote") ..."
        scp -q "${HPC_USER_HOST}:${remote}" "$dest"
    fi
}

# ---------------------------------------------------------------------------
# Prepare local tmp directory
# ---------------------------------------------------------------------------
mkdir -p "$LOCAL_TMP"

# ---------------------------------------------------------------------------
# Fetch the CSV
# ---------------------------------------------------------------------------
CSV_LOCAL="${LOCAL_TMP}/tsa_failed_tracts.csv"
echo "Fetching CSV from ${HPC_USER_HOST}:${CSV_REMOTE} ..."
scp -q "${HPC_USER_HOST}:${CSV_REMOTE}" "$CSV_LOCAL"

# ---------------------------------------------------------------------------
# Fetch the MNI template (once)
# ---------------------------------------------------------------------------
TEMPLATE_LOCAL="${LOCAL_TMP}/Mean3G.nii.gz"
fetch "$HPC_TEMPLATE" "$TEMPLATE_LOCAL"

# ---------------------------------------------------------------------------
# Parse CSV into a temp lookup file: "subject|tract1 tract2 ..."
# Works with bash 3 (no associative arrays needed).
# ---------------------------------------------------------------------------
# Header: subject_session,Ent_R__LC_R,LC_L__Ent_L,...   (value 1 = failed)

LOOKUP="${LOCAL_TMP}/subject_failed_tracts.txt"
> "$LOOKUP"

# Read header line separately, then data lines
{
    IFS=',' read -r header_line
    # Strip BOM if present
    header_line="${header_line#$'\xef\xbb\xbf'}"
    # Split header into positional array via set --
    IFS=',' read -ra col_names <<< "$header_line"

    while IFS=',' read -r line; do
        [[ -z "$line" ]] && continue
        IFS=',' read -ra fields <<< "$line"
        subj="${fields[0]}"
        failed_list=""
        for i in "${!fields[@]}"; do
            [[ $i -eq 0 ]] && continue
            if [[ "${fields[$i]}" == "1" ]]; then
                failed_list="${failed_list} ${col_names[$i]}"
            fi
        done
        if [[ -n "$failed_list" ]]; then
            echo "${subj}|${failed_list# }" >> "$LOOKUP"
        fi
    done
} < "$CSV_LOCAL"

total=$(wc -l < "$LOOKUP" | tr -d ' ')
echo "Found ${total} subjects with ≥1 failed tract."
echo ""

# ---------------------------------------------------------------------------
# Fetch all unique ROI masks upfront (deduplicated)
# ---------------------------------------------------------------------------
FETCHED_ROIS_FILE="${LOCAL_TMP}/.fetched_rois"
touch "$FETCHED_ROIS_FILE"

fetch_roi_if_needed() {
    local roi_name="$1"
    if ! grep -qxF "$roi_name" "$FETCHED_ROIS_FILE"; then
        local dest="${LOCAL_TMP}/roi_${roi_name}.nii.gz"
        fetch "${HPC_ROIS_DIR}/${roi_name}.nii.gz" "$dest" || \
            echo "  WARNING: ROI ${roi_name}.nii.gz not found on HPC, skipping."
        echo "$roi_name" >> "$FETCHED_ROIS_FILE"
    fi
}

# ---------------------------------------------------------------------------
# Main QC loop — read subject|tracts pairs from the lookup file
# ---------------------------------------------------------------------------
idx=0
while IFS='|' read -r subj failed_tracts; do
    [[ -z "$subj" ]] && continue
    idx=$(( idx + 1 ))

    echo "========================================================"
    echo "Subject ${idx}/${total}: ${subj}"
    echo "  Failed tracts: ${failed_tracts}"
    echo "========================================================"

    subj_dir="${LOCAL_TMP}/${subj}"
    mkdir -p "$subj_dir"

    # Fetch FA registered to MNI space
    fa_mni_remote="${HPC_DERIVED}/${subj}/dwi/reg3G/FA_nlin2Mean3G.nii.gz"
    fa_mni_local="${subj_dir}/FA_nlin2Mean3G.nii.gz"
    if ! fetch "$fa_mni_remote" "$fa_mni_local"; then
        echo "  WARNING: FA_nlin2Mean3G.nii.gz not found for ${subj}, skipping."
        continue
    fi

    # Fetch native FA (sanity check layer)
    fa_native_remote="${HPC_DERIVED}/${subj}/dwi/dti_FA.nii.gz"
    fa_native_local="${subj_dir}/dti_FA.nii.gz"
    fetch "$fa_native_remote" "$fa_native_local" || true

    # Collect unique ROI names for this subject's failed tracts
    roi_list=()
    roi_names=()
    seen_rois=""
    for tract in $failed_tracts; do
        roi_a="${tract%%__*}"
        roi_b="${tract##*__}"
        for roi in "$roi_a" "$roi_b"; do
            case " $seen_rois " in
                *" $roi "*) continue ;;
            esac
            seen_rois="$seen_rois $roi"
            fetch_roi_if_needed "$roi"
            roi_local="${LOCAL_TMP}/roi_${roi}.nii.gz"
            if [[ -f "$roi_local" ]]; then
                roi_list+=( "$roi_local" )
                roi_names+=( "$roi" )
            fi
        done
    done

    # -----------------------------------------------------------------------
    # Combine all ROI masks into a single labelled segmentation using nibabel.
    # Label N = roi_names[N-1], so ITK-SNAP colour-codes each ROI separately.
    # -----------------------------------------------------------------------
    combined_seg="${subj_dir}/rois_seg.nii.gz"
    if [[ ${#roi_list[@]} -gt 0 ]]; then
        # Build Python list literals
        py_files="["
        for f in "${roi_list[@]}"; do py_files="${py_files}'${f}',"; done
        py_files="${py_files}]"

        python3 - <<PYEOF
import nibabel as nib, numpy as np
rois = ${py_files}
ref  = nib.load(rois[0])
seg  = np.zeros(ref.shape, dtype=np.int16)
for label, path in enumerate(rois, start=1):
    data = nib.load(path).get_fdata()
    seg[data > 0.5] = label
nib.save(nib.Nifti1Image(seg, ref.affine), '${combined_seg}')
PYEOF
        echo "  ROI segmentation: ${roi_names[*]} (labels 1..${#roi_list[@]})"
    fi

    # -----------------------------------------------------------------------
    # Fetch LC_L and LC_R fdt_paths (native DWI space), sum, then warp to
    # MNI space using the subject's FA→Mean3G warp field so they can be
    # overlaid alongside FA_nlin2Mean3G in ITK-SNAP.
    # Requires applywarp (FSL) to be on the local PATH.
    # -----------------------------------------------------------------------
    fdt_native_sum="${subj_dir}/fdt_paths_LC_sum_native.nii.gz"
    fdt_mni_local="${subj_dir}/fdt_paths_LC_sum_mni.nii.gz"
    warp_local="${subj_dir}/FA_warp2Mean3G.nii.gz"

    fdt_fetched=()
    for hemi in LC_L LC_R; do
        fdt_remote="${HPC_DERIVED}/${subj}/dwi/probtrackX/LC-MTL/${hemi}/fdt_paths.nii.gz"
        fdt_local="${subj_dir}/fdt_paths_${hemi}.nii.gz"
        fetch "$fdt_remote" "$fdt_local" && fdt_fetched+=( "$fdt_local" ) || true
    done

    if [[ ${#fdt_fetched[@]} -gt 0 ]]; then
        # Sum in native space
        py_fdt="["
        for f in "${fdt_fetched[@]}"; do py_fdt="${py_fdt}'${f}',"; done
        py_fdt="${py_fdt}]"
        python3 - <<PYEOF
import nibabel as nib, numpy as np
paths = ${py_fdt}
ref   = nib.load(paths[0])
total = np.zeros(ref.shape, dtype=np.float32)
for p in paths:
    total += nib.load(p).get_fdata().astype(np.float32)
nib.save(nib.Nifti1Image(total, ref.affine), '${fdt_native_sum}')
PYEOF

        # Fetch warp field + affine premat, then warp fdt_paths into MNI space.
        # applywarp needs --premat for the linear component because FA_warp2Mean3G
        # contains only the nonlinear (fnirt) part.
        mat_local="${subj_dir}/FA_lin2Mean3G.mat"
        fetch "${HPC_DERIVED}/${subj}/dwi/reg3G/FA_warp2Mean3G.nii.gz" "$warp_local" || true
        fetch "${HPC_DERIVED}/${subj}/dwi/reg3G/FA_lin2Mean3G.mat"     "$mat_local"  || true

        if [[ -f "$warp_local" ]]; then
            if command -v applywarp &>/dev/null; then
                premat_arg=""
                [[ -f "$mat_local" ]] && premat_arg="--premat=${mat_local}"
                applywarp --in="$fdt_native_sum" \
                          --ref="$TEMPLATE_LOCAL" \
                          --warp="$warp_local" \
                          $premat_arg \
                          --out="$fdt_mni_local" \
                          --interp=trilinear 2>/dev/null \
                    && echo "  fdt_paths: warped to MNI → fdt_paths_LC_sum_mni.nii.gz" \
                    || echo "  WARNING: applywarp failed for fdt_paths"
            else
                echo "  WARNING: applywarp not found on PATH — skipping fdt_paths overlay"
                echo "           (install FSL locally or add it to PATH)"
            fi
        else
            echo "  WARNING: FA_warp2Mean3G.nii.gz not found — skipping fdt_paths overlay"
        fi
    fi

    # -----------------------------------------------------------------------
    # Launch ITK-SNAP in the background.
    #   -g  main greyscale:  MNI template (Mean3G)
    #   -o  image overlays:  FA registered to MNI, summed fdt_paths
    #   -s  segmentation:    combined labelled ROI mask
    # -----------------------------------------------------------------------
    snap_cmd=( "$ITKSNAP_BIN" -g "$TEMPLATE_LOCAL" -o "$fa_mni_local" )
    [[ -f "$fdt_mni_local" ]] && snap_cmd+=( -o "$fdt_mni_local" )
    [[ -f "$combined_seg" ]] && snap_cmd+=( -s "$combined_seg" )

    echo "  Launching ITK-SNAP ..."
    "${snap_cmd[@]}" >/dev/null 2>/dev/null &
    SNAP_PID=$!
    sleep 2   # let the GUI appear before showing the prompt

    printf "\n  Press Enter for next subject, or q+Enter to quit: "
    read -r key < /dev/tty
    echo ""

    if [[ "$key" == "q" || "$key" == "Q" ]]; then
        kill "$SNAP_PID" 2>/dev/null || true
        wait "$SNAP_PID" 2>/dev/null || true
        echo "Quitting."
        exit 0
    fi


    # Close ITK-SNAP (suppress "Terminated" noise) and delete subject files
    kill "$SNAP_PID" 2>/dev/null || true
    wait "$SNAP_PID" 2>/dev/null || true
    rm -rf "$subj_dir"
    echo "  Deleted local files for ${subj}."
    echo ""
done < <(sort "$LOOKUP")

echo "QC complete."
