#!/bin/bash
# Submit probtrackx LC-native network jobs for a list of subjects
# Usage: ./queue_probtrackx_native.sh [subjects_file]
# Default subjects file: project/test_subjects_20.list

SUBJECTS_FILE="${1:-/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project/test_subjects_20.list}"
CONFIG="/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project/config_preprocess_native.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while read -r subject; do
    sbatch "${SCRIPT_DIR}/probtrackx_gpu_job.sh" "$subject" "$CONFIG"
done < "$SUBJECTS_FILE"
