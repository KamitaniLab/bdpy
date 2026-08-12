#!/bin/bash
# Download the pre-processed ds006319 test fixture from figshare, verify its
# checksum, and place it under tests/data/mri/ for the real-data test
# (tests/mri/fmriprep/test_fmriprep_real.py).
#
# This replaces the datalad + FreeSurfer + fMRIPrep pipeline that previously had
# to be run locally. Those scripts are kept, for reproducing the fixture itself,
# in ../fixture_generation/ and are not needed to run the test.
#
# Fixture: https://doi.org/10.6084/m9.figshare.32857559.v1 (CC BY 4.0)
#   ds006319_fmriprep-1.2.1_minimal.tar.gz, 1,769,324,197 bytes, ~2.9 GB unpacked
#
# Extracted layout:
#   <dest>/ds006319/                     minimal BIDS + fMRIPrep 1.2.1 subset
#   <dest>/golden_master/real/           expected BData output + label mapper
#
# Usage:
#   bash ./tests/mri/fmriprep/scripts/real/step_1_figshare_download.sh [dest_dir]
#
# Environment:
#   FIGSHARE_FILE_ID        figshare file id (default: 67295762)
#   FIGSHARE_MD5            expected md5 of the archive (default: the fixture above)
#   FIGSHARE_LOCAL_TARBALL  use a local archive instead of downloading, for
#                           offline runs or for checking a regenerated fixture
set -euo pipefail

DEST="${1:-./tests/data/mri}"
FILE_ID="${FIGSHARE_FILE_ID:-67295762}"
EXPECTED_MD5="${FIGSHARE_MD5:-7f4ec67fb6e73239d3e5dc1066ef8f55}"
LOCAL_TARBALL="${FIGSHARE_LOCAL_TARBALL:-}"

dataset_dir="${DEST}/ds006319"
expected_h5="${DEST}/golden_master/real/test_output_fmriprep_real_exclude.h5"

if [ -d "${dataset_dir}" ] && [ -f "${expected_h5}" ]; then
  echo "Fixture already present under ${DEST}; nothing to do."
  echo "Remove ${dataset_dir} and ${expected_h5} to force a re-download."
  exit 0
fi

mkdir -p "${DEST}"
archive="${DEST}/.figshare_fixture.tar.gz"

if [ -n "${LOCAL_TARBALL}" ]; then
  echo "Using local archive: ${LOCAL_TARBALL}"
  [ -f "${LOCAL_TARBALL}" ] || { echo "Local archive not found: ${LOCAL_TARBALL}"; exit 1; }
  cp "${LOCAL_TARBALL}" "${archive}"
else
  url="https://ndownloader.figshare.com/files/${FILE_ID}"
  echo "Downloading fixture from figshare: ${url}"
  if command -v curl > /dev/null 2>&1; then
    curl -fL --retry 3 -C - -o "${archive}" "${url}"
  elif command -v wget > /dev/null 2>&1; then
    wget -c -O "${archive}" "${url}"
  else
    echo "Neither curl nor wget is available."
    exit 1
  fi
fi

echo "Verifying checksum..."
if command -v md5sum > /dev/null 2>&1; then
  actual_md5="$(md5sum "${archive}" | cut -d' ' -f1)"
elif command -v md5 > /dev/null 2>&1; then
  actual_md5="$(md5 -q "${archive}")"
else
  echo "Neither md5sum nor md5 is available; cannot verify the archive."
  exit 1
fi

if [ "${actual_md5}" != "${EXPECTED_MD5}" ]; then
  echo "Checksum mismatch. The download is incomplete or the fixture has changed."
  echo "  expected: ${EXPECTED_MD5}"
  echo "  actual  : ${actual_md5}"
  echo "Delete ${archive} and run this script again."
  exit 1
fi
echo "md5 OK: ${actual_md5}"

echo "Extracting into ${DEST} ..."
tar xzf "${archive}" -C "${DEST}"
rm -f "${archive}"

[ -d "${dataset_dir}" ] || { echo "Expected directory missing after extraction: ${dataset_dir}"; exit 1; }
[ -f "${expected_h5}" ] || { echo "Expected file missing after extraction: ${expected_h5}"; exit 1; }

echo "----------------------------------------"
echo "Fixture ready:"
echo "  ${dataset_dir}"
echo "  ${DEST}/golden_master/real"
echo "Next: bash ./tests/mri/fmriprep/scripts/real/step_2_run_test.sh"
echo "----------------------------------------"
