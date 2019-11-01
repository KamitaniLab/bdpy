# BdPy

Python package for brain decoding analysis

## Requirements

- Python 2 (mainly developed with Python 2.7.12)
    - Python 3 support is upcoming.
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

- bdata: BdPy data format (Bdata)
- dataform: Data format
- distcomp: Distributed computation
- fig: Making figures
- ml: Machine learning
- mri: MRI
- preproc: Preprocessing
- stats: Statistics
- util: Miscellaneous utilities

## BdPy data format

BdPy data format (or BrainDecoderToolbox2 data format; Bdata) consists of two variables: dataset and metadata. **dataset** stores brain activity data (e.g., voxel signal value for fMRI data), target variables (e.g., ID of stimuli for vision experiments), and additional information specifying experimental design (e.g., run and block numbers for fMRI experiments). Each row corresponds to a single 'sample', and each column representes either single feature (voxel), target, or experiment design information. **metadata** contains data describing meta-information for each column in dataset.

### Data API

#### Import module and initialization.

    from bdpy import Bdata

    # Create an empty Bdata instance
    dat = BData()

    # Load Bdata from a file
    dat = BData('data_file.h5')

#### Load data

    # Load Bdata from 'data_file.h5'
    dat.load('data_file.h5')

#### Show data

    # Show 'key' and 'description' of metadata
    dat.show_meatadata()

    # Get 'value' of the metadata specified by 'key'
    voxel_x = dat.get_metadata('voxel_x', where='VoxelData')

#### Data extraction

    # Get an array of voxel data in V1
    data_v1 = dat.select('ROI_V1')  # shape=(M, num voxels in V1)

    # `select` accepts some operators
    data_v1v2 = dat.select('ROI_V1 + ROI_V2')
    data_hvc = data.select('ROI_LOC + ROI_FFA + ROI_PPA - LOC_LVC')

    # Wildcard
    data_visual = data.select('ROI_V*')

    # Get labels ('image_index') in the dataset
    label_a  = dat.select('image_index')

#### Data creation

    # Add new data
    x = numpy.random.rand(dat.dataset.shape[0])
    dat.add(x, 'random_data')

    # Set description of metadata
    dat.set_metadatadescription('random_data', 'Random data')

    # Save data
    dat.save('output_file.h5')  # File format is selected automatically by extension. .mat, .h5,and .npy are supported.

## For developers

Please send your pull requests to `dev` branch, not to `master`.

## Contributors

- Shuntaro C. Aoki (Kyoto Univ)
- Misato Tanaka (DNI, ATR)
