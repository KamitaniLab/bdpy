
PYTHON_BIN="${PYTHON_BIN:-python}"

# Naming the marker opts in; tests/conftest.py deselects these otherwise.
"${PYTHON_BIN}" -m pytest -m real_data ./tests/mri/fmriprep/test_fmriprep_real.py
