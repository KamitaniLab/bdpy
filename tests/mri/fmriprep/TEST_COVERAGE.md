# Test Coverage and Maintenance Notes for `bdpy.mri.fmriprep`

This document summarizes what the current test suite covers, what remains untested, and known implementation issues observed during test reorganization.

_Last updated: 2026-08-13_

## Test Coverage Summary

`test_fmriprep_mock.py` covers:

- `FmriprepData` parsing and dataset traversal
- failure cases for missing or malformed inputs
- golden-master validation for `create_bdata_fmriprep`
- surface loading through `BrainData`
- `LabelMapper`
- subject-level BData creation
- `cut_run` behavior
- volume loading through `BrainData`

`test_fmriprep_real.py` covers:

- golden-master validation of `create_bdata_fmriprep` using real fMRIPrep output with exclusions

## Functions and Classes in `bdpy/mri/fmriprep.py`

### FmriprepData (class)

Parses a BIDS/fMRIPrep directory and stores subject/session/run structures in an `OrderedDict`.
Internally calls `__parse_data` (subject/session/run discovery) and `__get_task_event_files` (events.tsv + bold.json pairing).

| Item | Details |
| --- | --- |
| Test status | Covered |
| Real-data dependency | Replaced by mock directory structure |
| Existing tests | `TestFmriprepDataMock`: private vars, get_subjects, get_sessions, parse_session, parse_data, get_task_event_files / `TestFmriprepDataFailures` (parsing failures): missing events, missing bold.json |
| Not yet tested | File-pattern branch for fMRIPrep versions `1.0`/`1.1` |

Note: `TestFmriprepDataFailures` is a single test class containing 5 methods. Two (missing events, missing bold.json) test `FmriprepData` parsing failures; the remaining three (missing confounds, missing motion columns, unknown labels) test failures during `create_bdata_fmriprep` execution. They are listed under the respective sections below.

### create_bdata_fmriprep (function)

Top-level orchestrator. Parses with `FmriprepData`, applies exclusions (`subject`/`session`/`run`/`session-run`), loads `label_mapper` files (csv/tsv), and calls `__create_bdata_fmriprep_subject` per subject. Can also split outputs by task label when `split_task_label=True`.

| Item | Details |
| --- | --- |
| Test status | Partially covered; known issues remain |
| Real-data dependency | Replaced by mock data |
| Existing tests | `TestCreateBdataFmriprepMock`: golden masters (with/without exclusion for `volume_native`; `surface_native` with `with_confounds=True`), surface-standard shape variants, `split_task_label=True` for single-task mock data, empty list when subject excluded / `TestFmriprepDataFailures` (execution failures): missing confounds, missing motion columns, unknown labels |
| Gaps / issues | (1) Potential mutation-while-iterating issue: deleting from `fmriprep.data` (`OrderedDict`) during iteration may raise `RuntimeError: dictionary changed size during iteration` on Python 3.8. (2) `split_task_label=True` mock test exercises the branch with a single task only; the multi-task case (where `bdata_list` has multiple elements) is covered by `test_fmriprep_real.py`. Extending `MockBidsBuilder` for multi-task is a possible follow-up. (3) No focused unit test for csv/tsv `label_mapper` loading logic. (4) No explicit test for `return_list=False` single-BData return path. |

### BrainData (class)

Loads fMRI data (volume or surface).
For volume: reads NIfTI via `nipy.load_image`, reshapes 4D to 2D (sample x voxel), and computes xyz/ijk from affine.
For surface: loads left/right GIFTI via nibabel and concatenates.

| Item | Details |
| --- | --- |
| Test status | Surface path covered at the value level via `TestCreateBdataFmriprepMock.test_create_bdata_fmriprep_surface_native_gm`; volume path mostly indirect |
| Real-data dependency | Replaced by mock data |
| Existing tests | `TestBrainDataMock`: paired surface load, one-side missing, empty darray, vertex-count mismatch, invalid dtype |
| Gaps / issues | (1) No direct unit test for volume loader (`__load_volume`), only indirect coverage through `__create_bdata_fmriprep_subject`. (2) String identity checks (`is` / `is not`) are used for dtype and should be equality checks (`==` / `!=`). (3) No direct test for xyz/ijk extraction on 3D volume input. |

### LabelMapper (class)

Maps string labels to numeric values. Converts `'n/a'` to `NaN` and validates reverse-mapping uniqueness.

| Item | Details |
| --- | --- |
| Test status | Mostly covered |
| Real-data dependency | None |
| Existing tests | `TestLabelMapperMock` (3 methods): normal mapping including n/a→NaN (`test_get_value_returns_expected_numeric_mapping`), same-value mapping (`test_get_value_returns_existing_value_for_duplicate_mapping`), unknown key (`test_get_value_raises_on_missing_key`) |
| Not yet tested | (1) Non-unique reverse mapping case that raises `RuntimeError('Invalid label-value mapping')` (mapping different labels to the same value in separate calls, as opposed to the tested case where multiple labels share the same value in one dict). (2) Direct assertion of `dump()` output. |

### create_bdata_singlesubject (function)

Thin public wrapper for `__create_bdata_fmriprep_subject`; forwards arguments directly.

| Item | Details |
| --- | --- |
| Test status | Covered |
| Real-data dependency | Replaced by mock data |
| Existing tests | `TestCreateBdataFmriprepSubjectMock.test_create_bdata_singlesubject` |
| Not yet tested | Calling this public API with `cut_run=True` (the private function path is tested directly). |

### __create_bdata_fmriprep_subject (private function)

Core worker. For each run: loads with `BrainData`, reads confounds/motion, builds block/label arrays from task events, and writes to BData. Supports cropping when `cut_run=True` and event length exceeds data length.

| Item | Details |
| --- | --- |
| Test status | Covered |
| Real-data dependency | Replaced by mock data |
| Existing tests | golden master, direct call path, `cut_run` (True/False), confounds with NaN |
| Not yet tested | `NotImplementedError` path when `cut_duration < 0` |

### BrainData.__load_volume (private method)

Loads a NIfTI volume via `nibabel` and populates `data`, `xyz`, and `index` for 3D/4D input.

| Item | Details |
| --- | --- |
| Test status | Covered |
| Real-data dependency | Replaced by mock data |
| Existing tests | `TestBrainDataVolumeMock`: 4D mock, 3D temp file, invalid dimensions |
| Notes | Replaces the former module-level `__get_xyz` / `__load_mri` helpers, which upstream removed as unused in `ffa951f` (#143) after the nipy-to-nibabel migration (#138). |

## Main Findings

- `create_bdata_fmriprep` has a potential dictionary-mutation bug while iterating over `OrderedDict`.
- The `is`-based string comparisons in `BrainData` were fixed upstream in `ee9c42e` (#124).
- The module-level `__get_xyz` / `__load_mri` helpers were outside the main path and were removed upstream in `ffa951f` (#143); their coverage now targets `BrainData` directly.
- `split_task_label=True` is exercised by two mock tests: the single-task case
  (`test_create_bdata_fmriprep_split_task_label_single_task`) and the combination with
  `exclude` (`test_create_bdata_fmriprep_split_task_label_with_exclude`), which mirrors the
  parameters of the real-data test so that combination is covered without the multi-GB fixture.
  Multi-task mock coverage (multiple elements in `bdata_list`) would require extending `MockBidsBuilder`.
- Real-data tests (`test_fmriprep_real.py`) are marked with `pytest.mark.real_data` and are
  deselected by default via `addopts` in `pyproject.toml`; opt in with `pytest -m real_data`.
  Their volume-mode key list (`VOLUME_NATIVE_CHECK_KEYS` in `test_fmriprep_utils.py`) is shared
  with the mock tests, which extend it with the mock-only columns `image_index` and
  `original_run_number`.
- The real-data fixture is downloaded from figshare (doi:10.6084/m9.figshare.32857559.v1) by `scripts/real/step_1_figshare_download.sh` rather than regenerated locally, so running the real-data test no longer requires datalad, FreeSurfer, or Docker. The scripts that produce the fixture are kept in `scripts/fixture_generation/`.
