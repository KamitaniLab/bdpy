#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${script_dir}/_fs_env.sh" ]; then
  # Optional local-only overrides for machine-specific FreeSurfer paths.
  source "${script_dir}/_fs_env.sh"
fi

echo "Running fMRIPrep"

docker_image="poldracklab/fmriprep:1.2.1"

input_dir="${BIDS_DIR:-./tests/data/mri/ds006319}"
output_dir="${input_dir}/derivatives/fmriprep"
fs_subjects_dir="${FS_OUTPUT_DIR:-${input_dir}/derivatives/freesurfer}"

participant_label="S1"
fs_subject="sub-${participant_label}"

target_sessions=(
  "ses-SoundTest01"
  "ses-SoundTest02"
)

anat_session="ses-anatomy"

nthreads=32
omp_nthreads=8

FS_LICENSE="${FS_LICENSE:-}"
[ -n "${FS_LICENSE}" ] || { echo "Set FS_LICENSE before running this script."; exit 1; }

uid="$(id -u)"
gid="$(id -g)"

tmp_root="${output_dir}/tmp"
work_dir="${tmp_root}/work"
subset_dir="${tmp_root}/bids_subset"

check_broken_symlinks() {
  local src="$1"
  local broken

  broken="$(find -L "${src}" -type l -print)"
  if [ -n "${broken}" ]; then
    echo "ERROR: unresolved symlinks detected under ${src}"
    printf '%s\n' "${broken}"
    echo "This usually means the dataset is only partially materialized."
    echo "Run datalad get for the paths above, then rerun this script."
    exit 1
  fi
}

copy_with_rsync() {
  local src="$1"
  local dest="$2"

  check_broken_symlinks "${src}"
  if ! rsync -aL "${src}" "${dest}"; then
    echo "ERROR: rsync failed while copying ${src} -> ${dest}"
    echo "If the previous rsync output mentions missing files, rerun datalad get on that source path."
    exit 1
  fi
}

[ -f "${FS_LICENSE}" ] || { echo "FS license not found: ${FS_LICENSE}"; exit 1; }
[ -d "${input_dir}" ] || { echo "BIDS input_dir not found: ${input_dir}"; exit 1; }
[ -f "${input_dir}/dataset_description.json" ] || {
  echo "dataset_description.json not found in ${input_dir}"
  exit 1
}
[ -d "${fs_subjects_dir}/${fs_subject}" ] || {
  echo "FreeSurfer subject not found: ${fs_subjects_dir}/${fs_subject}"
  exit 1
}

for ses in "${target_sessions[@]}"; do
  [ -d "${input_dir}/sub-${participant_label}/${ses}" ] || {
    echo "Session not found: ${input_dir}/sub-${participant_label}/${ses}"
    exit 1
  }
done

[ -d "${input_dir}/sub-${participant_label}/${anat_session}" ] || {
  echo "Anatomy session not found: ${input_dir}/sub-${participant_label}/${anat_session}"
  exit 1
}

echo "All sanity checks passed. Proceeding with BIDS subset preparation..."

mkdir -p "${output_dir}"
rm -rf "${subset_dir}" "${work_dir}"
mkdir -p "${subset_dir}/sub-${participant_label}" "${work_dir}"

cp -f "${input_dir}/dataset_description.json" "${subset_dir}/"

for f in participants.tsv participants.json README CHANGES scans.json; do
  if [ -e "${input_dir}/${f}" ]; then
    copy_with_rsync "${input_dir}/${f}" "${subset_dir}/"
  fi
done

for ses in "${target_sessions[@]}"; do
  copy_with_rsync "${input_dir}/sub-${participant_label}/${ses}" "${subset_dir}/sub-${participant_label}/"
done

copy_with_rsync "${input_dir}/sub-${participant_label}/${anat_session}" "${subset_dir}/sub-${participant_label}/"

# Fail fast if any symlink remains in the subset
if find "${subset_dir}" -type l | grep -q .; then
  echo "ERROR: symlinks remain in ${subset_dir}"
  find "${subset_dir}" -type l
  exit 1
fi

mkdir -p "${output_dir}/freesurfer"

# Always sync FS subject to avoid stale partial copies
rsync -a "${fs_subjects_dir}/${fs_subject}/" "${output_dir}/freesurfer/${fs_subject}/"

if [ ! -f "${output_dir}/freesurfer/${fs_subject}/mri/aseg.presurf.mgz" ] && \
   [ -f "${output_dir}/freesurfer/${fs_subject}/mri/aseg.auto.mgz" ]; then
  cp -f "${output_dir}/freesurfer/${fs_subject}/mri/aseg.auto.mgz" \
        "${output_dir}/freesurfer/${fs_subject}/mri/aseg.presurf.mgz"
fi

docker run --rm \
  -u "${uid}:${gid}" \
  -v "${subset_dir}:/data:ro" \
  -v "${output_dir}:/out" \
  -v "${work_dir}:/work" \
  -v "${FS_LICENSE}:/opt/freesurfer/license.txt:ro" \
  "${docker_image}" \
  /data /out participant \
  --participant-label "${participant_label}" \
  --fs-license-file /opt/freesurfer/license.txt \
  -w /work \
  --nthreads "${nthreads}" \
  --omp-nthreads "${omp_nthreads}" \
  --output-space T1w template fsnative fsaverage fsaverage5 fsaverage6

echo "----------------------------------------"
echo "Preprocessing with fMRIPrep: completed"
echo "----------------------------------------"
