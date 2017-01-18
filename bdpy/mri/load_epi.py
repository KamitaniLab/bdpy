# coding: utf-8


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
    dataFiles: EPI image filepath list (.img)
    
    Returns
    -------
    data: 2D signal power matrix <sample * voxel>
    xyz_array: mni coordinate array <xyz * voxel>
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
    与えられたaffine行列に対して、全座標の組み合わせを生成

    Parameters
    ----------
    affine: imgファイルのaffine行列
    volume_shape: 1ボリュームにおける各次元のボクセルサイズのリスト(xyz voxel size)
    
    Returns
    -------
    xyz_array: 3 x n (number of voxels) matrix of XYZ locations returned (in mm) 
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
