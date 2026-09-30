#!/bin/bash
set -euo pipefail

data_root="./tests/data/mri"
dataset_name="ds006319"
dataset_dir="${data_root}/${dataset_name}"

mkdir -p "${data_root}"
cd "${data_root}"

if [ ! -d "${dataset_name}/.git" ]; then
  datalad install "https://github.com/OpenNeuroDatasets/${dataset_name}.git"
fi

cd "${dataset_name}"

# Materialize the exact tree consumed by the FreeSurfer/fMRIPrep helper scripts.
datalad get -r \
  dataset_description.json \
  participants.tsv \
  participants.json \
  README \
  CHANGES \
  sub-S1/ses-SoundTest01 \
  sub-S1/ses-SoundTest02 \
  sub-S1/ses-anatomy

echo "Download completed for ${dataset_dir}"
