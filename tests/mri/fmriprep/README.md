# Test code for `bdpy.mri.fmriprep`

> **Path note:** The canonical location is `tests/mri/fmriprep/` (not `tests/bdpy/mri`).

## Overview

This directory contains tests for `bdpy.mri.fmriprep`.

The layout is split by **which datasets a test runs against**, which is also the question
to ask when deciding where a new test goes:

| File | Runs against | Holds |
| --- | --- | --- |
| `test_both_datasets.py` | mock **and** real | Properties that must hold for every fMRIPrep dataset. The test bodies live here once and are driven from both fixtures, so adding one method extends both suites. Also the check that fails when production code grows a public function no test claims. |
| `test_mock_only.py` | mock | What only the synthetic dataset can express: exact comparisons against `MockBidsBuilder`'s independently rebuilt expectations, and inputs broken on purpose. |
| `test_real_only.py` | real | Comparison against the recorded 2.5 GB output. |

A test belongs in `test_both_datasets.py` if its assertion would still make sense against
a dataset nobody has seen yet. One that never touches a dataset at all — `BrainData`
rejecting an unknown dtype, say — is a unit test and belongs in `test_mock_only.py`.

Files whose names begin with `_` hold no tests and are not collected by pytest:

- `_support.py`: the shared key list, real-dataset paths, and `RealDatasetMixin`
- `_mock_fixtures.py`: `MockBidsBuilder`, the shared `DATA_BUILDER`, `MockDatasetMixin`, and
  the helper that rebuilds the expected BData without going through production code
- `scripts/mock/`: helper scripts for mock golden-master preparation and test execution
- `scripts/real/`: helper scripts for downloading the real-data fixture and running the real-data test
- `scripts/fixture_generation/`: scripts for regenerating the published fixture from the raw dataset; not needed to run the tests

### The two golden masters are not equally strong

Both suites compare against a stored `.h5`, but the stored value means different things,
which decides how much a failure tells you.

The **mock** ones are built by `build_expected_bdata_after_exclude`, which re-derives the
expectation from the source files with nibabel, independently of `create_bdata_fmriprep`.
A mismatch means production disagrees with an independent calculation.

The **real** one is a recording of `create_bdata_fmriprep`'s own output. Its only claim is
"this is what it did last time", so it pins current behaviour including any bug that
behaviour currently has. A mismatch means something changed, not that something broke.
This is why it is not called a ground truth: nothing outside the code under test attests
that those numbers are right.

For per-function test coverage, remaining gaps, and maintenance notes, see `TEST_COVERAGE.md`.

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

Run everything that needs no external fixture — `test_mock_only.py` plus the mock half
of `test_both_datasets.py`. The real-data tests are deselected automatically:

```bash
"${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/
```

Or use the helper scripts:

```bash
bash ./tests/mri/fmriprep/scripts/mock/step_1_prepare_gm.sh
bash ./tests/mri/fmriprep/scripts/mock/step_2_run_test.sh
```

Script roles:

- `step_1_prepare_gm.sh`: creates missing mock golden-master files if needed
- `step_2_run_test.sh`: runs every test under `tests/mri/fmriprep/` that needs no external fixture

Mock golden-master files:

- `./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5`
- `./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_exclude.h5`
- `./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_surface.h5`

If you need to regenerate them manually:

```bash
TEST_FMRIPREP_CREATE_GOLDEN_MASTER=1 "${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/test_mock_only.py
```

## Real-Data Tests

The real-data tests are marked with `pytest.mark.real_data`, and `tests/conftest.py`
deselects them unless the marker expression names `real_data`, so any ordinary
invocation — `pytest tests`, `pytest tests -m "not slow"` — skips over them.
They read a multi-GB fixture and take minutes, so opting in is explicit:

```bash
# every real-data test: the shared tests plus the stored-expectation comparison
"${PYTHON_BIN:-python}" -m pytest -m real_data ./tests/mri/fmriprep/
# or just the stored-expectation comparison
"${PYTHON_BIN:-python}" -m pytest -m real_data ./tests/mri/fmriprep/test_real_only.py
```

The `-m real_data` is required: without it the default deselection applies and
nothing is collected.

### Running only the shared tests

The shared tests in `test_both_datasets.py` assert properties of the dataset itself
rather than comparing against a recorded result, so they set
`requires_golden_master = False` and never read the 2.5 GB
`test_output_fmriprep_real_exclude.h5`. That is the loop worth reaching for while
working on the scanning code — about a minute instead of five:

```bash
"${PYTHON_BIN:-python}" -m pytest -m real_data ./tests/mri/fmriprep/test_both_datasets.py
```

The download is the same either way: the published fixture is a single archive holding
both the dataset and the stored expectation.

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
- `step_2_run_test.sh`: runs every `real_data` test under `tests/mri/fmriprep/`

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

The subset corresponds one-to-one with `exclude={"session/run": [[1, 2, 3, 4, 5], [1]]}` in `test_real_only.py`: `ses-SoundTest01` is fully excluded but must exist to occupy session index 0, `run-01` of `ses-SoundTest02` is excluded but is still needed for run discovery and event pairing, and `run-02` and `run-03` are the runs actually loaded. Changing `exclude`, the session layout, or `data_mode` therefore requires regenerating the fixture.

### Regenerating The Fixture

Only needed if the fixture itself has to change. The raw-dataset download, FreeSurfer, fMRIPrep, golden-master creation, and archive-building steps live in `scripts/fixture_generation/`; see `scripts/fixture_generation/README.md` for requirements and the step order.

## Notes

- The split `mock` / `real` structure replaced the older `USE_REAL`-based mixed test module.
- The entry points are `test_both_datasets.py`, `test_mock_only.py`, and `test_real_only.py`.
  Running `pytest ./tests/mri/fmriprep/` covers all three; naming a single file runs only
  part of the suite.
- The real-data helper scripts and tests prefer `./tests/data/mri/ds006319`. The test helpers also accept the older `./tests/data/ds006319` layout for compatibility.

## Submission Checklist

Before submitting, run this minimal checklist from the project root:

```bash
# 1) Everything that needs no external fixture: test_mock_only.py plus the
#    mock half of test_both_datasets.py. Real-data tests are deselected here.
"${PYTHON_BIN:-python}" -m pytest ./tests/mri/fmriprep/

# 2) Real-data tests (after step_1_figshare_download.sh has fetched the fixture)
"${PYTHON_BIN:-python}" -m pytest -m real_data ./tests/mri/fmriprep/

# 3) Verify golden-master files exist
test -f ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject.h5
test -f ./tests/data/mri/golden_master/mock/test_output_fmriprep_subject_exclude.h5
test -f ./tests/data/mri/golden_master/real/test_output_fmriprep_real_exclude.h5
```

Recommended for reproducibility:

- Keep machine-specific FreeSurfer settings in `scripts/fixture_generation/_fs_env.sh` (local-only). They are needed only when regenerating the fixture.
- Do not commit `_fs_env.sh`; only `_fs_env.sh.example` should be tracked.
- If golden masters were regenerated intentionally, mention that explicitly in your submission notes. Regenerating the real-data golden master means the published figshare fixture no longer matches, so it has to be rebuilt and republished as a new version.
