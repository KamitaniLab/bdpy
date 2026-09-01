# Tests for `bdpy.mri.fmriprep`

Tests for `bdpy/mri/fmriprep.py`. They come in two kinds: tests on synthetic brain data,
which `MockBidsBuilder` writes to a temporary directory, and tests on real brain data — a
minimal subset of an fMRIPrep-processed dataset published on figshare.

The synthetic ones need no data of their own. `pytest tests` from the project root runs
them along with the rest of the suite; to run this module's alone:

```bash
pytest ./tests/mri/fmriprep/
```

The real ones need that dataset downloaded first. They are left out of every ordinary
pytest run, so asking for them means naming their marker:

```bash
bash ./tests/mri/fmriprep/scripts/real/step_1_figshare_download.sh   # 1.7 GB, once
pytest -m real_data
```

Both assume bdpy is installed with its `mri` extra, are run from the project root, and
expect the directory layout of fMRIPrep 1.2. Adding `./tests/mri/fmriprep/test_both_datasets.py`
to the marker run selects the real-data tests that never read the 2.5 GB recorded output,
which is the faster loop while working on the scanning code.

## Test files

Which of the two datasets a test runs on decides the file it lives in:

| File | Runs against | Holds |
| --- | --- | --- |
| `test_both_datasets.py` | synthetic **and** real | Properties that must hold for every fMRIPrep dataset. The test bodies live here once and are driven from both datasets, so adding one method extends both suites. Also the check that fails when production code grows a public function no test claims. |
| `test_mock_only.py` | synthetic | What only the synthetic dataset can express: exact comparisons against `MockBidsBuilder`'s independently rebuilt expectations, and inputs broken on purpose. |
| `test_real_only.py` | real | Comparison against the recorded output. |

Supporting modules, prefixed with `_` so pytest does not collect them:

- `_support.py`: the shared key list, real-dataset paths, and `RealDatasetMixin`
- `_mock_fixtures.py`: `MockBidsBuilder`, the shared `DATA_BUILDER`, `MockDatasetMixin`, and
  the helper that rebuilds the expected BData without going through production code

`scripts/mock/` and `scripts/real/` wrap the commands above; `scripts/fixture_generation/`
holds the steps that produce the published dataset.

For per-function coverage, the remaining gaps, and maintenance notes, see
`TEST_COVERAGE.md`.

## The real dataset

fMRIPrep 1.2.1 output for one subject of OpenNeuro
[ds006319](https://openneuro.org/datasets/ds006319/versions/1.0.1) v1.0.1 (CC0), published
as [10.6084/m9.figshare.32857559.v1](https://doi.org/10.6084/m9.figshare.32857559.v1)
(CC BY 4.0). This fMRI dataset is collected in a standard experiment, in the fMRIPrep
layout, but it keeps only the minimum the tests involve. Some directories therefore hold
no files at all: the tests skip one session and one run, so their images were left out,
and sessions and runs are addressed by position, so the slots themselves had to stay.

```
tests/data/mri/
  ds006319/
    sub-S1/ses-SoundTest02/func/                      run-01..03 events.tsv and bold.json
    derivatives/fmriprep/fmriprep/sub-S1/
      ses-SoundTest01/func/                           empty, holds session index 0
      ses-SoundTest02/func/                           run-01 confounds; run-02 and run-03
                                                      space-T1w_desc-preproc_bold plus confounds
  golden_master/real/
    test_output_fmriprep_real_exclude.h5              recorded output
    temp.tsv                                          pinned stimulus_name label mapper
```

Regenerating and republishing the dataset is covered in
`scripts/fixture_generation/README.md`. To point the tests at a locally rebuilt archive,
set `FIGSHARE_LOCAL_TARBALL=/path/to/archive.tar.gz`.
