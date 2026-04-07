# Test code for `bdpy.mri.fmriprep`

> **Path note:** The canonical location is `tests/mri/fmriprep/` (not `tests/bdpy/mri`).

## Overview

This directory contains tests for `bdpy.mri.fmriprep`.

The current layout is split by data source:

- `test_fmriprep_mock.py`: mock-data tests that run without external datasets
- `test_fmriprep_real.py`: real-data golden-master test
- `test_fmriprep_utils.py`: shared helpers for loading `fmriprep.py` and for real-data tests
- `test_fmriprep_utils_mock.py`: mock dataset builder and expected-data helpers
- `scripts/mock/`: helper scripts for mock golden-master preparation and test execution
- `scripts/real/`: helper scripts for real-data download, preprocessing, golden-master preparation, and test execution

For test coverage and maintenance notes, see `TEST_COVERAGE.md`.

## Dependencies

- Run these tests in the lab base environment where `bdpy`, `nipy`, `nibabel`, and related dependencies are already installed.
- These tests assume the directory layout of `fmriprep` version `1.2`.
- `bdpy/mri/fmriprep.py` still depends on the older `nipy` API (`get_data()`), so newer environments may fail even if the test code itself is correct.
- Real-data tests additionally require `datalad` for dataset download and `docker` for `poldracklab/fmriprep:1.2.1`.

## Working Directory

Run all commands below from the project root, which is the directory containing `tests/` and `bdpy/`.

Set the Python interpreter explicitly when needed:

```bash
export PYTHON_BIN=/path/to/python
```

If `PYTHON_BIN` is unset, the helper scripts default to `python`.

## Mock Tests

Run the mock-only suite directly:

```bash
"${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_mock.py
```

Or use the helper scripts:

```bash
bash ./tests/mri/fmriprep/scripts/mock/step_1_prepare_gm.sh
bash ./tests/mri/fmriprep/scripts/mock/step_2_run_test.sh
```

Script roles:

- `step_1_prepare_gm.sh`: creates missing mock golden-master files if needed
- `step_2_run_test.sh`: runs `test_fmriprep_mock.py`

Mock golden-master files:

- `./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5`
- `./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_exclude.h5`

If you need to regenerate them manually:

```bash
TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1 "${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_mock.py
```

## Real-Data Tests

The real-data workflow is staged because it depends on external data and preprocessing outputs.

### Environment

For preprocessing scripts, set the environment variables below explicitly:

```bash
export BIDS_DIR=./tests/data/mri/ds006319
export FREESURFER_HOME=/path/to/freesurfer
export FS_LICENSE=/path/to/license.txt
export FS_OUTPUT_DIR="${BIDS_DIR}/derivatives/freesurfer"
```

To avoid committing machine-specific paths, copy the local template below and edit it on your machine only:

```bash
cp ./tests/mri/fmriprep/scripts/real/_fs_env.sh.example ./tests/mri/fmriprep/scripts/real/_fs_env.sh
```

`step_2_run_freesurfer.sh` and `step_3_run_fmriprep.sh` read `./tests/mri/fmriprep/scripts/real/_fs_env.sh` if it exists, then fall back to the current shell environment.
Keep `_fs_env.sh` local-only; the repository only tracks `_fs_env.sh.example`.

### Run The Full Pipeline

```bash
bash ./tests/mri/fmriprep/scripts/real/step_1_download.sh
bash ./tests/mri/fmriprep/scripts/real/step_2_run_freesurfer.sh
bash ./tests/mri/fmriprep/scripts/real/step_3_run_fmriprep.sh
bash ./tests/mri/fmriprep/scripts/real/step_4_prepare_gm.sh
bash ./tests/mri/fmriprep/scripts/real/step_5_run_test.sh
```

Script roles:

- `step_1_download.sh`: downloads the raw real dataset
- `step_2_run_freesurfer.sh`: runs FreeSurfer
- `step_3_run_fmriprep.sh`: runs fMRIPrep using the two functional sessions plus `ses-anatomy`
- `step_4_prepare_gm.sh`: creates the real-data golden master if it is missing
- `step_5_run_test.sh`: runs `test_fmriprep_real.py`

### Run The Test Only

If the dataset, FreeSurfer output, and fMRIPrep output are already prepared, run the test directly:

```bash
"${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_real.py
```

Real-data golden-master file:

- `./tests/data/mri/golden_master/real/test_output_fmriprep_real_exclude.h5`

If you need to regenerate it manually:

```bash
TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1 "${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_real.py
```

## Notes

- The split `mock` / `real` structure replaced the older `USE_REAL`-based mixed test module.
- The current recommended entry points are `test_fmriprep_mock.py` and `test_fmriprep_real.py`.
- The real-data helper scripts and tests prefer `./tests/data/mri/ds006319`. The test helpers also accept the older `./tests/data/ds006319` layout for compatibility.

## Submission Checklist

Before submitting, run this minimal checklist from the project root:

```bash
# 1) Mock tests
"${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_mock.py

# 2) Real-data tests (only when dataset + preprocessing outputs are ready)
"${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_real.py

# 3) Verify golden-master files exist
test -f ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5
test -f ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_exclude.h5
test -f ./tests/data/mri/golden_master/real/test_output_fmriprep_real_exclude.h5
```

Recommended for reproducibility:

- Keep machine-specific FreeSurfer settings in `scripts/real/_fs_env.sh` (local-only).
- Do not commit `_fs_env.sh`; only `_fs_env.sh.example` should be tracked.
- If golden masters were regenerated intentionally, mention that explicitly in your submission notes.
