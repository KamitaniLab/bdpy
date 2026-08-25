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
- `scripts/real/`: helper scripts for downloading the real-data fixture and running the real-data test
- `scripts/fixture_generation/`: scripts for regenerating the published fixture from the raw dataset; not needed to run the tests

For test coverage and maintenance notes, see `TEST_COVERAGE.md`.

## Dependencies

- Run these tests in the lab base environment where `bdpy`, `nipy`, `nibabel`, and related dependencies are already installed.
- These tests assume the directory layout of `fmriprep` version `1.2`.
- `bdpy/mri/fmriprep.py` itself now uses `nibabel` only, but `bdpy.mri` still imports `nipy` through `bdpy/mri/glm.py`, so `nipy` must be installed to import the package.
- Real-data tests additionally require `curl` (or `wget`) and `tar` to fetch the published fixture. `datalad`, FreeSurfer, and `docker` are needed only to regenerate that fixture; see `scripts/fixture_generation/README.md`.

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
- `./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_surface.h5`

If you need to regenerate them manually:

```bash
TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1 "${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_mock.py
```

## Real-Data Tests

The real-data tests are marked with `pytest.mark.real_data`, and `tests/conftest.py`
deselects them unless the marker expression names `real_data`, so any ordinary
invocation — `pytest tests`, `pytest tests -m "not slow"` — skips over them.
They read a multi-GB fixture and take minutes, so opting in is explicit:

```bash
"${PYTHON_BIN:-python}" -m pytest -m real_data ./tests/mri/fmriprep/test_fmriprep_real.py
# or, for every real-data test under this directory:
"${PYTHON_BIN:-python}" -m pytest -m real_data ./tests/mri/fmriprep/
```

The `-m real_data` is required: without it the default deselection applies and
nothing is collected.

### The quick subset

Not every real-data test needs the full fixture. The shared tests in
`test_fmriprep_invariants.py` assert properties of the dataset itself rather than
comparing against a recorded result, so they need only the 478 MB dataset — not the
2.5 GB `test_output_fmriprep_real_exclude.h5` — and finish in about 70 seconds instead
of minutes. They carry `real_data_quick` in addition to `real_data`:

```bash
"${PYTHON_BIN:-python}" -m pytest -m real_data_quick ./tests/mri/fmriprep/
```

Because they are also marked `real_data`, the conftest deselection still hides them from
a plain `pytest tests`, and `-m real_data` still runs them alongside everything else.

These are the tests worth reaching for while working on the scanning code: the synthetic
dataset uses numeric session names (`ses-01`) and gives every run a BOLD image, whereas the
real fixture uses `ses-SoundTest02`, ships an empty session directory, and ships one run
with events but no BOLD. Assumptions about naming and completeness only break here.

The test data is a small fMRIPrep-processed fixture published on figshare. Download it once, then run the test:

```bash
bash ./tests/mri/fmriprep/scripts/real/step_1_figshare_download.sh
bash ./tests/mri/fmriprep/scripts/real/step_2_run_test.sh
```

Script roles:

- `step_1_figshare_download.sh`: downloads the fixture, verifies its md5, and extracts it under `./tests/data/mri/`
- `step_2_run_test.sh`: runs `test_fmriprep_real.py` with `-m real_data`

The download is explicit and never happens during a normal `pytest` run, so
`pytest` stays offline by default. Once the fixture has been downloaded, the
`real_data` deselection is what keeps `pytest tests` fast.

If you opt in with `-m real_data` while the fixture is missing, the tests
**fail with a `FileNotFoundError` naming the download command** rather than
skipping: you asked for the real-data tests explicitly, so silently reporting
success would be misleading.

### Fixture

| Item | Value |
| --- | --- |
| Archive | `ds006319_fmriprep-1.2.1_minimal.tar.gz` (`tar.gz`, not `zip`) |
| Version | figshare item 32857559, version 1 |
| DOI | [10.6084/m9.figshare.32857559.v1](https://doi.org/10.6084/m9.figshare.32857559.v1), CC BY 4.0 |
| md5 | `7f4ec67fb6e73239d3e5dc1066ef8f55` |
| Size | 1,769,324,197 bytes, about 2.9 GB unpacked |
| Source dataset | OpenNeuro [ds006319](https://openneuro.org/datasets/ds006319/versions/1.0.1) v1.0.1 (CC0), doi:10.18112/openneuro.ds006319.v1.0.1 |
| fMRIPrep | 1.2.1, Docker image `poldracklab/fmriprep:1.2.1` |
| FreeSurfer | 7.4.1 (`freesurfer-linux-ubuntu20_x86_64-7.4.1-20230614-7eb8460`), `recon-all -all` on `sub-S1_ses-anatomy_T1w.nii.gz`, run outside the container and passed to fMRIPrep as `derivatives/freesurfer` |
| Generation command | `bash ./tests/mri/fmriprep/scripts/fixture_generation/step_5_build_minimal_bundle.sh`, after `step_1` to `step_4` in the same directory |

`step_1_figshare_download.sh` refuses to extract an archive whose md5 does not match, so a truncated download fails loudly instead of producing a confusing test failure. To point the test at a locally rebuilt archive instead of the published one, set `FIGSHARE_LOCAL_TARBALL=/path/to/archive.tar.gz`.

Expected layout after extraction:

```
tests/data/mri/
  ds006319/
    sub-S1/ses-SoundTest02/func/                      run-01..03 events.tsv and bold.json
    derivatives/fmriprep/fmriprep/sub-S1/
      ses-SoundTest01/func/                           empty, holds session index 0
      ses-SoundTest02/func/                           run-01 confounds; run-02 and run-03
                                                      space-T1w_desc-preproc_bold plus confounds
  golden_master/real/
    test_output_fmriprep_real_exclude.h5              expected output
    temp.tsv                                          pinned stimulus_name label mapper
```

The subset corresponds one-to-one with `exclude={"session/run": [[1, 2, 3, 4, 5], [1]]}` in `test_fmriprep_real.py`: `ses-SoundTest01` is fully excluded but must exist to occupy session index 0, `run-01` of `ses-SoundTest02` is excluded but is still needed for run discovery and event pairing, and `run-02` and `run-03` are the runs actually loaded. Changing `exclude`, the session layout, or `data_mode` therefore requires regenerating the fixture.

### Regenerating The Fixture

Only needed if the fixture itself has to change. The raw-dataset download, FreeSurfer, fMRIPrep, golden-master creation, and archive-building steps live in `scripts/fixture_generation/`; see `scripts/fixture_generation/README.md` for requirements and the step order.

## Notes

- The split `mock` / `real` structure replaced the older `USE_REAL`-based mixed test module.
- The current recommended entry points are `test_fmriprep_mock.py` and `test_fmriprep_real.py`.
- The real-data helper scripts and tests prefer `./tests/data/mri/ds006319`. The test helpers also accept the older `./tests/data/ds006319` layout for compatibility.

## Submission Checklist

Before submitting, run this minimal checklist from the project root:

```bash
# 1) Mock tests
"${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_fmriprep_mock.py

# 2) Real-data tests (after step_1_figshare_download.sh has fetched the fixture)
"${PYTHON_BIN:-python}" -m pytest -m real_data ./tests/mri/fmriprep/test_fmriprep_real.py

# 3) Verify golden-master files exist
test -f ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5
test -f ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_exclude.h5
test -f ./tests/data/mri/golden_master/real/test_output_fmriprep_real_exclude.h5
```

Recommended for reproducibility:

- Keep machine-specific FreeSurfer settings in `scripts/fixture_generation/_fs_env.sh` (local-only). They are needed only when regenerating the fixture.
- Do not commit `_fs_env.sh`; only `_fs_env.sh.example` should be tracked.
- If golden masters were regenerated intentionally, mention that explicitly in your submission notes. Regenerating the real-data golden master means the published figshare fixture no longer matches, so it has to be rebuilt and republished as a new version.
