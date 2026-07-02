#!/bin/bash
#SBATCH --job-name=fix_ptx_structure
#SBATCH --partition=imgcomputeq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/preprocess/logs/slurm-%j_fix_ptx_structure.out

# Fix probtrackx output directory structure for subjects that ran with the
# intermediate code that wrote to probtrackX/LC-limbic/{net_name}/{seed}/
# instead of the correct flat probtrackX/LC-limbic/{seed}/.
#
# For each net_name subdir found, merge its seed-level subdirs upward into
# the parent (probtrackX/LC-limbic/{seed}/), then remove the empty net dirs.

set -euo pipefail

DERIV=/gpfs01/imgshare/ConnLS/ADNI/derived
NETWORK=LC-limbic
FIXED=0
SKIPPED=0

echo "=== fix_probtrackx_structure ==="
echo "Scanning $DERIV for wrong-structure output..."
echo ""

for ptx_dir in "$DERIV"/*/dwi/probtrackX/"$NETWORK"/; do
    [ -d "$ptx_dir" ] || continue
    subj=$(echo "$ptx_dir" | awk -F'/' '{print $(NF-4)}')

    # Find net_name subdirs — identified by containing seed-level subdirs
    # (net names contain a hyphen, e.g. Ent_L-bst-ofc)
    mapfile -t net_dirs < <(find "$ptx_dir" -mindepth 1 -maxdepth 1 -type d \
                              | xargs -I{} basename {} \
                              | grep -E "^(Ent|LC)_[LR]-")

    if [ ${#net_dirs[@]} -eq 0 ]; then
        continue
    fi

    echo "[$subj]  fixing ${#net_dirs[@]} network dir(s): ${net_dirs[*]}"

    for net_name in "${net_dirs[@]}"; do
        net_path="${ptx_dir}${net_name}"

        # Each subdir of net_path is a seed dir
        for seed_path in "$net_path"/*/; do
            [ -d "$seed_path" ] || continue
            seed=$(basename "$seed_path")
            dest="${ptx_dir}${seed}"

            mkdir -p "$dest"

            # Move files; skip if destination already has the file
            find "$seed_path" -maxdepth 1 -type f | while read -r f; do
                fname=$(basename "$f")
                if [ -e "${dest}/${fname}" ]; then
                    echo "  SKIP (exists): $seed/$fname"
                else
                    mv "$f" "${dest}/${fname}"
                fi
            done

            # Move any subdirs (e.g. FreeTracking output dir)
            find "$seed_path" -mindepth 1 -maxdepth 1 -type d | while read -r d; do
                dname=$(basename "$d")
                if [ -e "${dest}/${dname}" ]; then
                    echo "  SKIP (exists): $seed/$dname/"
                else
                    mv "$d" "${dest}/${dname}"
                fi
            done

            rmdir "$seed_path" 2>/dev/null || true
        done

        # Remove the now-empty net_name dir
        rmdir "$net_path" 2>/dev/null \
            && echo "  removed $net_name/" \
            || echo "  WARNING: $net_name/ not empty after merge — check manually"
    done

    FIXED=$((FIXED + 1))
    echo ""
done

echo "Done. Fixed $FIXED subject(s)."
