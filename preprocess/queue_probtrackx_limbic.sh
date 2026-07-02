#!/bin/bash
# Submit probtrackx LC-limbic network jobs for a list of subjects
# Usage: ./queue_probtrackx_limbic.sh [subjects_file]
# Default subjects file: project/test_subjects_20.list

SUBJECTS_FILE="${1:-/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project/test_subjects_20.list}"
CONFIG="/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project/config_preprocess_limbic.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while read -r x; do
    sbatch "${SCRIPT_DIR}/probtrackx_gpu_job.sh" "$x" "$CONFIG"
done < "$SUBJECTS_FILE"
