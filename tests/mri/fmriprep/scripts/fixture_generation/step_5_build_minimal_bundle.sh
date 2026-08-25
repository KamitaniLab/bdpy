#!/bin/bash
# Build the minimal fixture archive that is published on figshare, from the full
# fMRIPrep outputs produced by step_1 through step_4.
#
# The full dataset is about 60 GB; the archive built here is 1.7 GB and contains
# only what tests/mri/fmriprep/test_real_only.py actually reads. See
# README.md in this directory for which files are kept and why.
#
# Output:
#   <out_dir>/ds006319/                    minimal BIDS + fMRIPrep subset
#   <out_dir>/golden_master/real/          expected BData output + temp.tsv
#   ds006319_fmriprep-1.2.1_minimal.tar.gz archive to upload
#
# Usage:
#   bash ./tests/mri/fmriprep/scripts/fixture_generation/step_5_build_minimal_bundle.sh [data_root] [out_dir]
#
# Environment:
#   INCLUDE_GM=0   skip copying the golden master (~2.4 GB); the published
#                  archive needs it, so leave this unset unless you are testing
#                  the layout
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${1:-./tests/data/mri}"
OUT="${2:-./tests/data/mri/figshare_bundle}"
INCLUDE_GM="${INCLUDE_GM:-1}"
ARCHIVE="ds006319_fmriprep-1.2.1_minimal.tar.gz"

ds_src="${DATA_ROOT}/ds006319"
fp_src="${ds_src}/derivatives/fmriprep/fmriprep/sub-S1/ses-SoundTest02/func"
raw_src="${ds_src}/sub-S1/ses-SoundTest02/func"
gm_src="${DATA_ROOT}/golden_master/real"
prefix="sub-S1_ses-SoundTest02_task-vggsoundtest"

ds_out="${OUT}/ds006319"
fp_out="${ds_out}/derivatives/fmriprep/fmriprep/sub-S1"
raw_out="${ds_out}/sub-S1/ses-SoundTest02/func"

[ -d "${fp_src}" ] || { echo "fMRIPrep outputs not found: ${fp_src}"; echo "Run step_1 through step_3 first."; exit 1; }

rm -rf "${OUT}"
mkdir -p "${fp_out}/ses-SoundTest01/func"   # intentionally empty: holds session index 0
mkdir -p "${fp_out}/ses-SoundTest02/func"
mkdir -p "${raw_out}"
mkdir -p "${OUT}/golden_master/real"

# run-01 is excluded by the test, but its confounds file is still needed for run discovery.
cp -a "${fp_src}/${prefix}_run-01_desc-confounds_regressors.tsv" "${fp_out}/ses-SoundTest02/func/"

# run-02 and run-03 are the runs the test loads.
for run in 02 03; do
  cp -a "${fp_src}/${prefix}_run-${run}_space-T1w_desc-preproc_bold.nii.gz" "${fp_out}/ses-SoundTest02/func/"
  cp -a "${fp_src}/${prefix}_run-${run}_desc-confounds_regressors.tsv"      "${fp_out}/ses-SoundTest02/func/"
done

# Raw events and bold.json: fMRIPrep does not emit these, and the module reads
# them for stimulus labels and TR. run-01 is kept so the event-file glob matches.
for run in 01 02 03; do
  cp -a "${raw_src}/${prefix}_run-${run}_events.tsv" "${raw_out}/"
  cp -a "${raw_src}/${prefix}_run-${run}_bold.json"  "${raw_out}/"
done

touch "${fp_out}/ses-SoundTest01/func/.gitkeep"

# Pinned label mapper, so the test does not need all 560 raw events.tsv files.
if [ -f "${gm_src}/temp.tsv" ]; then
  cp -a "${gm_src}/temp.tsv" "${OUT}/golden_master/real/temp.tsv"
else
  echo "temp.tsv not found; generating it from the full dataset ..."
  "${PYTHON_BIN}" "${script_dir}/gen_label_mapper.py" "${ds_src}" "${OUT}/golden_master/real/temp.tsv"
fi

if [ "${INCLUDE_GM}" = "1" ]; then
  gm_file="${gm_src}/test_output_fmriprep_real_exclude.h5"
  [ -f "${gm_file}" ] || { echo "Golden master not found: ${gm_file}"; echo "Run step_4 first."; exit 1; }
  cp -a "${gm_file}" "${OUT}/golden_master/real/"
  # BData.save() writes the call stack, including absolute paths, into /header.
  echo "Stripping the BData header from the golden master ..."
  "${PYTHON_BIN}" "${script_dir}/clean_bdata_header.py" "${OUT}/golden_master/real/test_output_fmriprep_real_exclude.h5"
fi

echo "Creating ${ARCHIVE} ..."
tar czf "${ARCHIVE}" -C "${OUT}" .

echo "----------------------------------------"
echo "Bundle directory: ${OUT}"
du -sh "${OUT}"/* 2>/dev/null || true
echo "Archive: ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1))"
echo "md5: $(md5sum "${ARCHIVE}" 2>/dev/null | cut -d' ' -f1 || md5 -q "${ARCHIVE}")"
echo "Check the archive for local paths before publishing it."
echo "----------------------------------------"
