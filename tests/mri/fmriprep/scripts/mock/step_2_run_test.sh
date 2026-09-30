
PYTHON_BIN="${PYTHON_BIN:-python}"

# The whole directory, not test_mock_only.py alone: the mock half of
# test_both_datasets.py and the API coverage check live there too.
"${PYTHON_BIN}" -m pytest ./tests/mri/fmriprep/
