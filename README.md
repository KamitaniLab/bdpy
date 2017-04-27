# BdPy

Python library for brain decoding

## Requirements

- numpy
- scipy
- h5py

### Optional requirements

- 'dataform' module
    - pandas
- 'mri' module
    - nipy

## Modules

- bdata
- dataform
- ml
- mri
- preproc
- util

## BrainDecoderToolbox2 data format specification

BrainDecoderToolbox2 data format (bdata) consists of two variables: dataSet and metaData. 'dataSet' stores feature data (e.g., voxel signal value for fMRI data), targets (e.g., ID of stimuli for vision experiments), and additional information specifying experimental design (e.g., run and block numbers for fMRI experiments). Each row corresponds to a single 'sample', and each column representes either feature, target, or experiment design data. 'metaData' contains data describing meta-information for each column in dataSet.

- dataSet
    - 'dataSet' is a set of features, target vectors, and additional numeric data.
    - Each row in dataSet is a single sample
    - Each column in dataSet is either (1) a feature vector, (2) a label (target) vector, or (3) a vector specifying data design information (e.g., run number, block number, ...)
- metaData
    - 'metaData' is a set of values describing information on each column in 'dataSet'. Each meta-data is specified by a 'key' and has 'description'.

Example of metaData:

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

### Usage

##### Import module and initialization.

	from bdpy import BData
    
    # initialize and data load
    bd = BData()         # create an instance
An instance of the class 'BData' contains `dataSet` and `metaData` as instance variables. The 'dataSet' is a M x N numpy array (M: the number of samples, N: the number of all features which include brain data, labels, design information, etc).


##### data load
	
    bd.load( 'dataFile.h5' )  # 
##### Show Data

	bd.show_meatadata()                  # show 'key' and 'description' of the betaData
    voxel_x = bd.get_metaData('voxel_x') # get 'value' of the metaData specified by the 'key'.

##### Data extraction

    v1_data = bd.get_dataset('ROI_V1')  # get < M x "number of voxels which belong to V1" > array of voxel data.
	labelA  = bd.get_dataset('LabelA')  # 

##### Data creation

	bd.add_dataset( numpy.random.rand( bd.dataSet.shape[0]), 'random_feature' ) # add new data 
	bd.set_metadatadescription('random_feature', 'random data (just for test)') # set description of metaData

	# save
    bd.save( 'outputFile.h5' )  # File format is selected automatically by extension. .mat, .h5,and .npy are supported.
    
## For developers

Please send your pull requests to `dev` branch, not to `master`.

## Contributors

- Shuntaro C. Aoki (DNI, ATR)
- Misato Tanaka (DNI, ATR)
