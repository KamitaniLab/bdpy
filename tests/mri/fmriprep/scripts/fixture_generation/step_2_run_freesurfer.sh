#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${script_dir}/_fs_env.sh" ]; then
  # Optional local-only overrides for machine-specific FreeSurfer paths.
  source "${script_dir}/_fs_env.sh"
fi

BIDS_DIR="${BIDS_DIR:-./tests/data/mri/ds006319}"
FS_OUTPUT_DIR="${FS_OUTPUT_DIR:-${BIDS_DIR}/derivatives/freesurfer}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"

[ -n "${FREESURFER_HOME}" ] || { echo "Set FREESURFER_HOME before running this script."; exit 1; }
[ -d "${FREESURFER_HOME}" ] || { echo "FREESURFER_HOME not found: ${FREESURFER_HOME}"; exit 1; }

export SUBJECTS_DIR="${FS_OUTPUT_DIR}"
export FUNCTIONALS_DIR="${FUNCTIONALS_DIR:-${FREESURFER_HOME}/sessions}"
export NO_FSFAST=1
export OMP_NUM_THREADS
export FS_OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${OMP_NUM_THREADS}"

# FreeSurfer's setup script dereferences some unset variables.
set +u
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
set -u

t1w_file="${BIDS_DIR}/sub-S1/ses-anatomy/anat/sub-S1_ses-anatomy_T1w.nii.gz"
[ -f "${t1w_file}" ] || { echo "T1w file not found: ${t1w_file}"; exit 1; }

mkdir -p "${FS_OUTPUT_DIR}"
echo "Running recon-all for sub-S1 with ${OMP_NUM_THREADS} threads"

recon-all \
  -i "${t1w_file}" \
  -s sub-S1 \
  -all \
  -sd "${FS_OUTPUT_DIR}" \
  -openmp "${OMP_NUM_THREADS}" \
  -parallel
