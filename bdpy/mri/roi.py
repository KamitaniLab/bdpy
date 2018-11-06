"""
Utilities for ROIs
"""

import os

import numpy as np

from bdpy.mri import load_mri


def add_roimask(bdata, roi_mask, roi_prefix='',
                brain_data='VoxelData', xyz=['voxel_x', 'voxel_y', 'voxel_z'],
                return_roi_flag=False,
                verbose=True):
    '''Add an ROI mask to `bdata`.

    Parameters
    ----------
    bdata : BData
    roi_mask : str or list
        ROI mask file(s).

    Returns
    -------
    bdata : BData
    '''

    if isinstance(roi_mask, str):
        roi_mask = [roi_mask]
    
    # Get voxel xyz coordinates in `bdata`
    voxel_xyz = np.vstack([bdata.get_metadata(xyz[0], where=brain_data),
                           bdata.get_metadata(xyz[1], where=brain_data),
                           bdata.get_metadata(xyz[2], where=brain_data)])

    # Load the ROI mask files
    mask_xyz_all = []
    mask_v_all = []

    voxel_consistency = True
    
    for m in roi_mask:
        mask_v, mask_xyz, mask_ijk = load_mri(m)
        mask_v_all.append(mask_v)
        mask_xyz_all.append(mask_xyz[:, (mask_v == 1).flatten()])

        if not (voxel_xyz == mask_xyz).all():
            voxel_consistency = False

    # Get ROI flags
    if voxel_consistency:
        roi_flag = np.vstack(mask_v_all)
    else:
        roi_flag = get_roiflag(mask_xyz_all, voxel_xyz)

    # Add the ROI flag as metadata in `bdata`
    for i, roi in enumerate(roi_mask):
        roi_name = roi_prefix + '_' + os.path.basename(roi).split('.')[0]

        print('Adding %s' % roi_name)
        bdata.add_metadata(roi_name, roi_flag[i, :], description='1 = ROI %s' % roi_name, where=brain_data)

    if return_roi_flag:
        return bdata, roi_flag
    else:
        return bdata


def get_roiflag(roi_xyz_list, epi_xyz_array, verbose=True):
    """
    Get ROI flags

    Parameters
    ----------
    roi_xyz_list : list
        List of arrays that contain XYZ coordinates of ROIs. Each element is an
        array (3 * num of voxels in the ROI).
    epi_xyz_array : array
        Voxel XYZ coordinates (3 * num of voxels)
    verbose : boolean
        If True, 'get_roiflag' outputs verbose message

    Returns
    -------
    roi_flag : array
        ROI flags (Num of ROIs * num of voxels)
    """

    epi_voxel_size = epi_xyz_array.shape[1]
    print "EPI voxel size %d" % (epi_voxel_size)

    roi_flag_array = np.zeros((len(roi_xyz_list), epi_voxel_size))

    epi_xyz_array_t = np.transpose(epi_xyz_array)

    for i in range(len(roi_xyz_list)):
        print "getROIflag: ROI%d (Voxel Size %d)" % (i+1, len(roi_xyz_list[i][0]))

        # get roi's voxel size
        roi_xyz_num = len(roi_xyz_list[i][0])
        # transpose roi xyz list
        roi_xyz_array_t = np.transpose(roi_xyz_list[i])

        # limit epi's matching range
        range_list = []
        for j in range(3):
            # get min and max value every axis (x,y,z)
            # get true/false array if epi xyz between the min and max value
            min_val = min(roi_xyz_list[i][j,:])
            max_val = max(roi_xyz_list[i][j,:])
            min_range = epi_xyz_array[j,:] >= min_val
            max_range = epi_xyz_array[j,:] <= max_val
            range_list.append(np.asarray(min_range, dtype=int))
            range_list.append(np.asarray(max_range, dtype=int))
        # all range conditions is clear
        used_index_array = np.sum(np.asarray(range_list), axis = 0) == 6
        # get epi's index_list in the limited range
        used_epi_index_list = np.where(used_index_array == True)[0]

        # judge epi coordinates in a roi coordinates
        for j in range(len(used_epi_index_list)):
            index = used_epi_index_list[j]
            epi_xyz = epi_xyz_array_t[index]
            if np.any([np.array_equal(epi_xyz, roi_xyz_array_t[k]) for k in range(roi_xyz_num)]):
                roi_flag_array[i][index] = 1

    return roi_flag_array
