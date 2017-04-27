# BrainDecoderToolbox2

Matlab library for brain decoding

## Installation

- Put `BrainDecoderToolbox2` in your computer and run `setpath` on MATLAB.
- BrainDecoderToolbox2 requires SPM (5, 8, or 12).

## Files

- /bdata
    - Set of data interface functions for BrainDecoderToolbox2 data format
- /preprocessor
    - Collection of functions for preprocessing
- /util
    - Collection of utility functions
- /figure
    - Collection of functions for drawing figures
- /test
    - Test scripts
- setpath.m
    - Matlab script to set paths to BrainDecoderToolbox2 functions
- README.md
    - This file
- Makefile
    - Makefile
- .gitignore
    - Git related file

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
    | Label          | 1: the column is a label vector        | ( NaN, NaN, NaN, ...,   1,   1, ..., NaN, NaN ) |
    | Label_A        | 1: the column represents 'Label A'     | ( NaN, NaN, NaN, ...,   1,   0, ..., NaN, NaN ) |
    | ...            | ...                                    | ...                                             |
    | Block          | 1: the column represents block numbers | ( NaN, NaN, NaN, ..., NaN, NaN, ...,   1, NaN ) |
    | Run            | 1: the column represents run numbers   | ( NaN, NaN, NaN, ..., NaN, NaN, ..., NaN,   1 ) |
    +----------------+----------------------------------------+-------------------------------------------------+

### Implementation

- dataSet is a M x N matrix (M: the number of samples, N: the number of features + labels + design information + ...)
- metaData is a structure containing the following fields:
    - key         : cell array (length: T, T is the number of meta data)
    - description : cell array (length: T)
    - value       : matrix (T x N)

## For developers

- Functions should have names in snake-case.
- Variables should have names in lower camel-case.

## Third-party functions

BrainDecoderToolbox2 contains the following third-party functions:

- errorbar_h by The MathWorks, Inc.
- hline by Brandon Kuczenski
- suptitle by Drea Thomas, John Cristion, and Mark Histed

## Contributors

- Shuntaro C. Aoki (ATR)
