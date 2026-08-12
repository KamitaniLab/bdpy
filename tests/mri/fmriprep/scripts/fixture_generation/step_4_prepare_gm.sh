PYTHON_BIN="${PYTHON_BIN:-python}"

real_gm_h5_path=./tests/data/mri/golden_master/real/test_output_fmriprep_real_exclude.h5

if [ -f "${real_gm_h5_path}" ]; then
  echo "Real-data golden master file already exists: ${real_gm_h5_path}"
  echo "Skipping GM creation step. If you want to recreate the GM, please delete the existing file and run this script again."
  exit 0
fi

echo "Missing golden master file: ${real_gm_h5_path}"
TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1 "${PYTHON_BIN}" -m pytest ./tests/mri/fmriprep/test_fmriprep_real.py
