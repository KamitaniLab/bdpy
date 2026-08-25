
PYTHON_BIN="${PYTHON_BIN:-python}"

# -m real_data overrides the default deselection set in pyproject.toml.
"${PYTHON_BIN}" -m pytest -m real_data ./tests/mri/fmriprep/test_fmriprep_real.py
