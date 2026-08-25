
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pytest ./tests/mri/fmriprep/test_mock_only.py
