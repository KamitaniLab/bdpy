"""
Load EPIs
"""

import itertools as itr
import os
import re
import string

import nipy
import numpy as np
import scipy.io as sio


def load_epi(dataFiles):
    """
    Loads EPIs

    Parameters
    ----------
    dataFiles: list
        EPI image filepath list
    
    Returns
    -------
    data: array_like
        EPI voxel values (M * N; N is the number of samples, M is the nubmer of
        voxels)
    xyz_array: array_like
        Coordiantes of voxels (3 * N)
    """
    
    # load the first volume
    print "Loading %s" % (dataFiles[0])
    data = nipy.load_image(dataFiles[0])
    img = data.get_data()
    base_coordmap = data.coordmap
    
    # flatten 3d to 1d (decompose z->y->x)
    data_list = [np.array(img.flatten())]

    # make xyz coordinate
    xyz_array = _get_xyz(base_coordmap.affine, img.shape)
    
    for i in range(1, len(dataFiles)):
        print "Loading %s" % (dataFiles[i])
        
        # load data
        data = nipy.load_image(dataFiles[i])
        img = data.get_data()
        coordmap = data.coordmap
        
        # confirm that coordmap is same
        if base_coordmap != coordmap:
            raise NameError("invalid coordmap")
            break;
        
        # append flatten img to data_list 
        data_list.append(np.array(img.flatten(), dtype=np.float64))
    
    # convert list to array stacking vertically
    data_array = np.vstack(data_list)
        
    return data_array, xyz_array


def _get_xyz(affine, volume_shape):
    """
    Returns voxel xyz coordinates based on an affine matrix

    Parameters
    ----------
    affine : array
        Affine matrix
    volume_shape : array
        Size in x-, y-, and z-axes of each voxel
    
    Returns
    -------
    xyz_array : array
        x-, y-, and z-coordinates of each voxel (3 * N; N is the number of
        voxels)
    """
    xyz_list = []
    for i in [0,1,2]:
        # calculate coordinate every dim from affine matrix
        xyz_list.append([affine[i,i] * j + affine[i,3] for j in range(volume_shape[i])])
    # convert 3D to 2D (decompose voxels in order of z, y, and x)
    xyz_product_list = list(itr.product(xyz_list[0], xyz_list[1], xyz_list[2]))

    # convert list to array and transpose
    xyz_array = np.array(xyz_product_list).T
    
    return xyz_array
