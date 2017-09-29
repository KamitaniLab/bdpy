"""
Collection of misc utils

This file is a part of BdPy
"""


__all__ = ['get_refdata', 'makedir_ifnot']


import os

import numpy as np


def get_refdata(data, ref_key, foreign_key):
    """Get data referred by `foreign_key`

    Parameters
    ----------
    data : array_like
        Data array
    ref_key
        Reference keys for `data`
    foreign_key
        Foreign keys referring `data` via `ref_key`

    Returns
    -------
    array_like : array_like
        Referred data
    """

    ind = [np.where(ref_key == i)[0][0] for i in foreign_key]

    if data.ndim == 1:
        return data[ind]
    else:
        return data[ind, :]


def makedir_ifnot(dir_path):
    '''Make a directory if it does not exist

    Parameters
    ----------
    dir_path : str
        Path to the directory to be created

    Returns
    -------
    bool
        True if the directory was created
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        return True
    else:
        return False
