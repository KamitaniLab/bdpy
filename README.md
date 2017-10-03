# BdPy

Python library for brain decoding

## Requirements

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

Run the following command:

``` shell
$ pip install git+https://github.com/KamitaniLab/bdpy.git
```

or

``` shell
$ git clone git@github.com:KamitaniLab/bdpy.git
$ cd bdpy/
$ python setup.py install
```

## Modules

- bdata
- dataform
- ml
- mri
- preproc
- util

## BrainDecoderToolbox2/BdPy data format

BrainDecoderToolbox2/BdPy data format (bdata) consists of two variables: dataset and metadata. 'dataset' stores feature data (e.g., voxel signal value for fMRI data), targets (e.g., ID of stimuli for vision experiments), and additional information specifying experimental design (e.g., run and block numbers for fMRI experiments). Each row corresponds to a single 'sample', and each column representes either feature, target, or experiment design data. 'metadata' contains data describing meta-information for each column in dataset.

- dataset
    - 'dataset' is a set of features, target vectors, and additional numeric data.
    - Each row in dataset is a single sample
    - Each column in dataset is either (1) a feature vector, (2) a label (target) vector, or (3) a vector specifying data design information (e.g., run number, block number, ...)
- metadata
    - 'metadata' is a set of values describing information on each column in 'dataset'. Each meta-data is specified by a 'key' and has 'description'.

Example of metadata:

    +----------------+----------------------------------------+-------------------------------------------------+
    | key            | description                            | value                                           |
    +----------------+----------------------------------------+-------------------------------------------------+
    | Feature        | 1: the column is a feature vector      | (   1,   1,   1, ..., NaN, NaN, ..., NaN, NaN ) |
    | voxel_x        | Voxel x coordinate                     | (  24, 102,  28, ..., NaN, NaN, ..., NaN, NaN ) |
    | ...            | ...                                    | ...                                             |
    | ROI_V1         | 1: V1 data                             | (   0,   1,   1, ..., NaN, NaN, ..., NaN, NaN ) |
    | ...            | ...                                    | ...                                             |
    | Label          | 1: the column is a label vector        | ( NaN, NaN, NaN, ...,   1,   1, ..., NaN, NaN ) |
    | Label_A        | 1: the column represents 'Label A'     | ( NaN, NaN, NaN, ...,   1,   0, ..., NaN, NaN ) |
    | ...            | ...                                    | ...                                             |
    | Block          | 1: the column represents block numbers | ( NaN, NaN, NaN, ..., NaN, NaN, ...,   1, NaN ) |
    | Run            | 1: the column represents run numbers   | ( NaN, NaN, NaN, ..., NaN, NaN, ..., NaN,   1 ) |
    +----------------+----------------------------------------+-------------------------------------------------+

### Data API usage

#### Import module and initialization.

    from bdpy import BData

    # Create a BData instance
    bd = BData()

An instance of the class 'BData' contains `dataset` and `metadata` as instance variables. The 'dataset' is a M x N numpy array (M: the number of samples, N: the number of all features which include brain data, labels, design information, etc).

#### Load data

    # Load BData from 'data_file.h5'
    bd.load('data_file.h5')

#### Show Data

    # Show 'key' and 'description' of the betaData
    bd.show_meatadata()
    # Get 'value' of the metadata specified by the 'key'
    voxel_x = bd.get_metadata('voxel_x')

#### Data extraction

    # Get <M x "number of voxels which belong to V1"> array of voxel data
    v1_data = bd.get('ROI_V1')

    # another way to select data
    v1_data = bd.select('ROI_V1 = 1')

    # Get labels ('Label_A') in the dataset
    label_a  = bd.get('Label_A')

#### Data creation

    # Add new data
    bd.add(numpy.random.rand(bd.dataset.shape[0]), 'random_data')

    # Set description of metadata
    bd.set_metadatadescription('random_data', 'Random data (just for test)')

    # Save data
    bd.save('output_file.h5')  # File format is selected automatically by extension. .mat, .h5,and .npy are supported.

## For developers

Please send your pull requests to `dev` branch, not to `master`.

## Contributors

- Shuntaro C. Aoki (DNI, ATR)
- Misato Tanaka (DNI, ATR)
