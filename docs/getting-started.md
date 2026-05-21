# Getting Started

## Installation

Install the latest stable release from PyPI:

```shell
pip install bdpy
```

To install the latest development version:

```shell
pip install git+https://github.com/KamitaniLab/bdpy.git
```

### Optional dependencies

Some modules require additional packages:

```shell
# Deep learning (PyTorch)
pip install bdpy[torch]

# MRI utilities
pip install bdpy[mri]

# Figure utilities
pip install bdpy[fig]

# Pipeline support
pip install bdpy[pipeline]

# All optional dependencies
pip install bdpy[all]
```

## Basic usage

### Loading and working with BData

`BData` is the core data container in BdPy. It stores brain activity (e.g., fMRI voxel signals) together with metadata describing the experimental design.

```python
from bdpy import BData

# Load data from an HDF5 file
bdata = BData('data_file.h5')

# Show available metadata keys
bdata.show_metadata()

# Extract voxel data from a region of interest
data_v1 = bdata.select('ROI_V1')       # shape: (n_samples, n_voxels_in_V1)

# Combine ROIs using operators
data_v1v2 = bdata.select('ROI_V1 + ROI_V2')
data_hvc  = bdata.select('ROI_LOC + ROI_FFA + ROI_PPA - LOC_LVC')

# Wildcard selection
data_visual = bdata.select('ROI_V*')

# Extract stimulus labels
labels = bdata.select('image_index')
```

### Saving data

```python
import numpy as np

# Add a new data column
x = np.random.rand(bdata.dataset.shape[0])
bdata.add(x, 'random_data')
bdata.set_metadatadescription('random_data', 'Random data (example)')

# Save to HDF5
bdata.save('output_file.h5')
```

## Next steps

- See the [API Reference](api/index.md) for detailed documentation of all modules.
