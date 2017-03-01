"""
Utilities for ROIs
"""


import numpy as np


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
