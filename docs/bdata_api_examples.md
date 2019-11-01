# BData API examples

### Data API

#### Import module and initialization.

    from bdpy import BData

    # Create an empty BData instance
    dat = BData()

    # Load BData from a file
    dat = BData('data_file.h5')

#### Load data

    # Load BData from 'data_file.h5'
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
