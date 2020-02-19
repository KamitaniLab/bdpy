# BdPy

[![PyPI version](https://badge.fury.io/py/bdpy.svg)](https://badge.fury.io/py/bdpy)
[![GitHub license](https://img.shields.io/github/license/KamitaniLab/bdpy)](https://github.com/KamitaniLab/bdpy/blob/master/LICENSE)

Python package for brain decoding analysis

## Requirements

- Python 2.7, 3.6, or later
- numpy
- scipy
- scikit-learn
- h5py

### Optional requirements

- 'dataform' module
    - pandas
- 'mri' module
    - nipy

## Installation

``` shell
$ pip install bdpy
```

## Packages

- bdata: BdPy data format (BData)
- dataform: Data format
- distcomp: Distributed computation
- dl: Deep learning
- feature: DNN features
- fig: Making figures
- ml: Machine learning
- mri: MRI
- opendata: Open data
- preproc: Preprocessing
- stats: Statistics
- util: Miscellaneous utilities

## BdPy data format

BdPy data format (or BrainDecoderToolbox2 data format; BData) consists of two variables: dataset and metadata. **dataset** stores brain activity data (e.g., voxel signal value for fMRI data), target variables (e.g., ID of stimuli for vision experiments), and additional information specifying experimental design (e.g., run and block numbers for fMRI experiments). Each row corresponds to a single 'sample', and each column representes either single feature (voxel), target, or experiment design information. **metadata** contains data describing meta-information for each column in dataset.

See [BData API examples](https://github.com/KamitaniLab/bdpy/blob/master/docs/bdata_api_examples.md) for useage of BData.

## For developers

Please send your pull requests to `dev` branch, not to `master`.

## Contributors

- Shuntaro C. Aoki (Kyoto Univ)
- Misato Tanaka (DNI, ATR)
