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

## Packages

- bdata: BdPy data format
- dataform: Third-party data format (e.g., Pandas)
- distcomp: Distributed computation package
- fig: Mmaking figures
- ml: Machine learning
- mri: MRI
- preproc: Preprocessing
- stats: Statistics
- util: Miscellaneous utilities

## BdPy data format

BdPy data format (or BrainDecoderToolbox2 data format; bdata) consists of two variables: dataset and metadata. 'dataset' stores brain activity data (e.g., voxel signal value for fMRI data), targets (e.g., ID of stimuli for vision experiments), and additional information specifying experimental design (e.g., run and block numbers for fMRI experiments). Each row corresponds to a single 'sample', and each column representes either single feature (voxel), target, or experiment design information. 'metadata' contains data describing meta-information for each column in dataset.

### Data API usage

#### Import module and initialization.

    from bdpy import BData

    # Create a BData instance
    dat = BData()

An instance of the class 'BData' contains `dataset` and `metadata` as instance variables. `dataset` is a M x N numpy array (M: the number of samples, N: the number of all features (voxels), labels, design information, etc).

#### Load data

    # Load BData from 'data_file.h5'
    dat.load('data_file.h5')

#### Show data

    # Show 'key' and 'description' of the betaData
    dat.show_meatadata()

    # Get 'value' of the metadata specified by the 'key'
    voxel_x = dat.get_metadata('voxel_x', where='VoxelData')

#### Data extraction

    # Get an array of voxel data in V1 (shape = (M, [no. of voxels in V1]))
    v1_data = dat.select('ROI_V1')

    # Another way to select data
    v1_data = dat.select('ROI_V1 = 1')

    # Select data in several ROIs
    data_v1v2 = dat.select('ROI_V1 = 1 | ROI_V2 = 1')

    # Get labels ('Label_A') in the dataset
    label_a  = dat.select('Label_A')

#### Data creation

    # Add new data
    dat.add(numpy.random.rand(dat.dataset.shape[0]), 'random_data')

    # Set description of metadata
    dat.set_metadatadescription('random_data', 'Random data (just for test)')

    # Save data
    dat.save('output_file.h5')  # File format is selected automatically by extension. .mat, .h5,and .npy are supported.

## For developers

Please send your pull requests to `dev` branch, not to `master`.

## Contributors

- Shuntaro C. Aoki (DNI, ATR)
- Misato Tanaka (DNI, ATR)
