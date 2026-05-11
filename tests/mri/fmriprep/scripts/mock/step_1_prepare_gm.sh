PYTHON_BIN="${PYTHON_BIN:-python}"

mock_gm_h5_path_list=(
  ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_exclude.h5
  ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5
  ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_surface.h5
)

need_mock_gm=0

for gm_h5_path in "${mock_gm_h5_path_list[@]}"; do
  if [ ! -f "${gm_h5_path}" ]; then
    need_mock_gm=1
    echo "Missing golden master file: ${gm_h5_path}"
  fi
done

if [ "${need_mock_gm}" -eq 0 ]; then
  echo "All mock golden master files already exist."
  echo "Skipping GM creation step. If you want to recreate the GM, please delete the existing file and run this script again."
  exit 0
fi

TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1 "${PYTHON_BIN}" -m pytest ./tests/mri/fmriprep/test_fmriprep_mock.py
