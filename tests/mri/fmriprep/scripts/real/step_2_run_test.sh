
PYTHON_BIN="${PYTHON_BIN:-python}"

# Naming the marker opts in; tests/conftest.py deselects these otherwise.
# The whole directory, not test_real_only.py alone: the real half of
# test_both_datasets.py carries the real_data marker too.
"${PYTHON_BIN}" -m pytest -m real_data ./tests/mri/fmriprep/
