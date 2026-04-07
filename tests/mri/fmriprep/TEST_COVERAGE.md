# Test Coverage and Maintenance Notes for `bdpy.mri.fmriprep`

This document summarizes what the current test suite covers, what remains untested, and known implementation issues observed during test reorganization.

_Last updated: 2026-04-06_

## Test Coverage Summary

`test_fmriprep_mock.py` covers:

- `FmriprepData` parsing and dataset traversal
- failure cases for missing or malformed inputs
- golden-master validation for `create_bdata_fmriprep`
- surface loading through `BrainData`
- `LabelMapper`
- subject-level BData creation
- `cut_run` behavior
- `get_xyz`
- `load_mri`

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
| Existing tests | `TestCreateBdataFmriprepMock`: golden masters (with/without exclusion), surface modes, empty list when subject excluded / `TestFmriprepDataFailures` (execution failures): missing confounds, missing motion columns, unknown labels |
| Gaps / issues | (1) Potential mutation-while-iterating issue: deleting from `fmriprep.data` (`OrderedDict`) during iteration may raise `RuntimeError: dictionary changed size during iteration` on Python 3.8. (2) No direct test for `split_task_label=True`. (3) No focused unit test for csv/tsv `label_mapper` loading logic. (4) No explicit test for `return_list=False` single-BData return path. |

### BrainData (class)

Loads fMRI data (volume or surface).
For volume: reads NIfTI via `nipy.load_image`, reshapes 4D to 2D (sample x voxel), and computes xyz/ijk from affine.
For surface: loads left/right GIFTI via nibabel and concatenates.

| Item | Details |
| --- | --- |
| Test status | Surface path covered; volume path mostly indirect |
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

### __get_xyz (private function)

Computes voxel coordinates with `itertools.product` and `xrange`; supports 3D/4D images.

| Item | Details |
| --- | --- |
| Test status | Covered (with patching) |
| Real-data dependency | None (uses fake image objects) |
| Existing tests | `TestGetXyzMock`: 4D/3D fake images |
| Notes | Python 3 does not provide `xrange`, so tests patch `fmriprep.xrange = range`. The function is not part of the current main execution path, so maintenance priority is low. |

### __load_mri (private function)

Loads NIfTI via `nipy.load_image` and returns `(data, xyz, ijk)` for 3D/4D input. Its behavior is largely duplicated by `BrainData.__load_volume`.

| Item | Details |
| --- | --- |
| Test status | Covered |
| Real-data dependency | Replaced by mock data |
| Existing tests | `TestLoadMriMock`: 4D mock, 3D temp file, invalid dimensions |
| Notes | Code is largely duplicated with `BrainData.__load_volume`; retained for compatibility checks but a candidate for future cleanup. |

## Main Findings

- `create_bdata_fmriprep` has a potential dictionary-mutation bug while iterating over `OrderedDict`.
- `BrainData` contains `is`-based string comparisons that should be equality comparisons.
- `__get_xyz` and `__load_mri` appear to be outside the current main path.
- `split_task_label=True` still has no direct test coverage.
