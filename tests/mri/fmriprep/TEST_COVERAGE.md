# Test Coverage and Maintenance Notes for `bdpy.mri.fmriprep`

This document summarizes what the current test suite covers, what remains untested, and known implementation issues observed during test reorganization.

_Last updated: 2026-08-26_

## Test Coverage Summary

The suite is split three ways by what a test needs in order to be meaningful.

`test_both_datasets.py` holds the test bodies that assert properties true of
**any** fMRIPrep dataset, and drives them from both fixtures. One class per production
class or function; the two concrete classes at the bottom supply only a dataset location
and its expected values, so adding one method there extends the synthetic and the
real-data suites at once. Covers:

- `FmriprepData` traversal: subject/session discovery, the run key set, events/bold.json resolution
- `LabelMapper`: the dataset's own mapper covers every label its events use; `n/a` maps to NaN; `dump` inverts what was queried
- `BrainData`: volume `data`/`xyz`/`index` agree on voxel count; surface concatenates both hemispheres
- `create_bdata_singlesubject`: one run yields a BData whose rows and columns match the source image
- `create_bdata_fmriprep`: excluding every subject short-circuits before any volume is read
- an API coverage check that fails when production code grows a function no test claims

`test_mock_only.py` holds what only the synthetic dataset can express — exact
comparisons against `MockBidsBuilder`'s expected values, and inputs that must be broken
on purpose:

- `__parse_session` / `__parse_data` compared against `DATA_BUILDER.expected_runs`
- failure cases for missing or malformed inputs
- golden-master validation for `create_bdata_fmriprep` (volume, exclude, surface)
- `split_task_label` behaviour
- `LabelMapper` unit behaviour (duplicate values, missing key)
- surface loading edge cases through `BrainData`, `cut_run` behaviour, volume loading

`test_real_only.py` holds the stored-expectation comparison:

- golden-master validation of `create_bdata_fmriprep` using real fMRIPrep output with exclusions

### Running them

| Command | What runs | Needs |
| --- | --- | --- |
| `pytest tests` | synthetic only; every real-data test is deselected | nothing |
| `pytest -m real_data tests/mri/fmriprep/test_both_datasets.py` | the shared tests against the real fixture, ~60 s | the fixture, but not the 2.5 GB h5 |
| `pytest -m real_data` | the above plus the stored-expectation comparison, minutes | the dataset **and** the 2.5 GB h5 |

## Functions and Classes in `bdpy/mri/fmriprep.py`

### FmriprepData (class)

Parses a BIDS/fMRIPrep directory and stores subject/session/run structures in an `OrderedDict`.
Internally calls `__parse_data` (subject/session/run discovery) and `__get_task_event_files` (events.tsv + bold.json pairing).

| Item | Details |
| --- | --- |
| Test status | Covered |
| Real-data dependency | Replaced by mock directory structure |
| Existing tests | `FmriprepDataInvariants` (both fixtures): config fields, get_subjects, get_sessions, subject/session keying, the run key set, events/bold.json existence / `TestFmriprepDataMock`: parse_session and parse_data against `DATA_BUILDER.expected_runs` / `TestFmriprepDataFailures` (parsing failures): missing events, missing bold.json |
| Not yet tested | File-pattern branch for fMRIPrep versions `1.0`/`1.1` |

Note: `TestFmriprepDataFailures` is a single test class containing 5 methods. Two (missing events, missing bold.json) test `FmriprepData` parsing failures; the remaining three (missing confounds, missing motion columns, unknown labels) test failures during `create_bdata_fmriprep` execution. They are listed under the respective sections below.

### create_bdata_fmriprep (function)

Top-level orchestrator. Parses with `FmriprepData`, applies exclusions (`subject`/`session`/`run`/`session-run`), loads `label_mapper` files (csv/tsv), and calls `__create_bdata_fmriprep_subject` per subject. Can also split outputs by task label when `split_task_label=True`.

| Item | Details |
| --- | --- |
| Test status | Partially covered; known issues remain |
| Real-data dependency | Replaced by mock data |
| Existing tests | `TestCreateBdataFmriprepMock`: golden masters (with/without exclusion for `volume_native`; `surface_native` with `with_confounds=True`), surface-standard shape variants, `split_task_label=True` for single-task mock data, empty list when subject excluded / `TestExcludeMultipleSubjectsMock`: exclusion against a two-subject tree, including the `xfail` that reproduces the mutation bug below / `TestFmriprepDataFailures` (execution failures): missing confounds, missing motion columns, unknown labels |
| Gaps / issues | (1) Mutation-while-iterating bug, reported upstream as issue #125 and **reproduced** by `TestExcludeMultipleSubjectsMock.test_excluding_a_non_final_subject`: deleting from `fmriprep.data` (`OrderedDict`) while iterating it raises `RuntimeError: OrderedDict mutated during iteration` whenever the excluded subject is not the last key. The test is marked `xfail(strict=True)`, so it turns into an XPASS — and therefore a failure — the moment #125 is fixed, which is the signal to replace it with a plain assertion. (2) `split_task_label=True` is exercised with a single task only. The multi-task case (where `bdata_list` has multiple elements) is covered **nowhere**. `test_real_only.py` does pass `split_task_label=True`, but asserts `len(bdata_list) == 1`: the figshare fixture carries a single task (`task-vggsoundtest`), so the earlier claim that it covered the multi-task case is not correct. Left this way on purpose — splitting by task is not how the module is normally used here, so the published fixture was not built for it. If it is wanted, `MockBidsBuilder` is the cheaper side: a second task there costs regenerating the three mock golden masters and no download. (3) No focused unit test for csv/tsv `label_mapper` loading logic. (4) No explicit test for `return_list=False` single-BData return path. |

### BrainData (class)

Loads fMRI data (volume or surface).
For volume: reads NIfTI via `nibabel.load` / `get_fdata`, reshapes 4D to 2D (sample x voxel), and computes xyz/ijk from affine.
For surface: loads left/right GIFTI via nibabel and concatenates.

| Item | Details |
| --- | --- |
| Test status | Covered directly on both fixtures |
| Real-data dependency | Replaced by mock data |
| Existing tests | `BrainDataInvariants` (both fixtures): volume `data`/`xyz`/`index` voxel-count agreement; surface hemisphere concatenation (synthetic only — see below) / `TestBrainDataMock`: paired surface load, one-side missing, empty darray, vertex-count mismatch, invalid dtype / `TestBrainDataVolumeMock`: 4D, 3D, invalid dimensions |
| Gaps / issues | Surface loading never runs against real data: the ds006319 fixture ships no `*.gii`, so `BrainDataInvariants` skips that test there with a message saying so. Recorded in `UNCOVERED_BEHAVIOUR`. |

### LabelMapper (class)

Maps string labels to numeric values. Converts `'n/a'` to `NaN` and validates reverse-mapping uniqueness.

| Item | Details |
| --- | --- |
| Test status | Mostly covered |
| Real-data dependency | None |
| Existing tests | `LabelMapperInvariants` (both fixtures): the dataset's own mapper covers every label its events use, `n/a`→NaN, `dump()` inverts the labels queried / `TestLabelMapperMock` (3 methods): same-value mapping, unknown key, and the basic numeric mapping |
| Not yet tested | Non-unique reverse mapping that raises `RuntimeError('Invalid label-value mapping')` — different labels resolving to one value across separate calls, as opposed to the tested case where several labels share a value in one dict. |

### create_bdata_singlesubject (function)

Thin public wrapper for `__create_bdata_fmriprep_subject`; forwards arguments directly.

| Item | Details |
| --- | --- |
| Test status | Covered |
| Real-data dependency | Replaced by mock data |
| Existing tests | `SingleSubjectInvariants` (both fixtures): one run yields a BData whose row counts match across `VoxelData`/`Session`/`Run`/`Block`/`Confounds` and whose voxel count matches the run's own image |
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

- `create_bdata_fmriprep` mutates the `OrderedDict` it is iterating over. Reported upstream as
  issue #125 and reproduced here by `TestExcludeMultipleSubjectsMock.test_excluding_a_non_final_subject`,
  which is marked `xfail(strict=True)` so that fixing #125 makes it fail loudly rather than pass
  unnoticed. Not fixed in this PR: it is production code, and the fix belongs with the issue.
- The `is`-based string comparisons in `BrainData` were fixed upstream in `ee9c42e` (#124).
- The module-level `__get_xyz` / `__load_mri` helpers were outside the main path and were removed upstream in `ffa951f` (#143); their coverage now targets `BrainData` directly.
- `split_task_label=True` is exercised by two mock tests: the single-task case
  (`test_create_bdata_fmriprep_split_task_label_single_task`) and the combination with
  `exclude` (`test_create_bdata_fmriprep_split_task_label_with_exclude`), which mirrors the
  parameters of the real-data test so that combination is covered without the multi-GB fixture.
- **Multi-task `split_task_label` is covered nowhere.** Earlier notes claimed the real-data
  test covered it; that is not so. `test_real_only.py` does pass `split_task_label=True`, so
  the branch runs, but it asserts `len(bdata_list) == 1` — the figshare fixture carries a
  single task (`task-vggsoundtest`), as does the mock dataset, so neither reaches the
  multi-element return. This is deliberate: splitting by task is not how the module is
  normally used here, so the published real-data fixture was not built for it. If the
  coverage is wanted, `MockBidsBuilder` is the cheaper side to add it on — a second task
  there costs regenerating the three mock golden-master files and no download. Recorded in
  `UNCOVERED_BEHAVIOUR` in `test_both_datasets.py`.
- Real-data tests (`test_real_only.py`) are marked with `pytest.mark.real_data` and are
  deselected by default in `tests/conftest.py`; opt in with `pytest -m real_data`.
  Their volume-mode key list (`VOLUME_NATIVE_CHECK_KEYS` in `_support.py`) is shared
  with the mock tests, which extend it with the mock-only columns `image_index` and
  `original_run_number`.
- The real-data fixture is downloaded from figshare (doi:10.6084/m9.figshare.32857559.v1) by `scripts/real/step_1_figshare_download.sh` rather than regenerated locally, so running the real-data test no longer requires datalad, FreeSurfer, or Docker. The scripts that produce the fixture are kept in `scripts/fixture_generation/`.
