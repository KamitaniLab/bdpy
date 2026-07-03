"""Sparse array class.

This file is a part of bdpy.
"""

__all__ = ['SparseArray', 'load_array', 'save_array', 'save_multiarrays']


import os
import warnings

import h5py
import hdf5storage
import numpy as np

from . import _mat_v73

# hdf5storage.savemat can fail for SparseArray.save under NumPy 2.x, especially
# when overwriting an existing sparse struct. SparseArray.save therefore uses a
# direct h5py writer only under NumPy >= 2. Dense-array save paths are unchanged
# and still rely on hdf5storage for MATLAB-v7.3 metadata/compatibility behavior.
_NUMPY2 = int(np.__version__.split('.')[0]) >= 2

# Deprecation notice emitted by the MATLAB-compatible write paths. The actual
# switch to bdpy-native plain HDF5 (and the drop of the hdf5storage write
# dependency) is implemented on the refactor/drop-hdf5storage-write branch.
_MATLAB_WRITE_FUTURE_WARNING = (
    "Writing MATLAB-compatible v7.3 .mat files is deprecated and will change "
    "in a future release: bdpy will write bdpy-native plain HDF5 instead. "
    "Newly written files will no longer be guaranteed to be readable by "
    "MATLAB's load(). Reading existing hdf5storage / MATLAB v7.3 files remains "
    "supported."
)


def load_array(fname, key='data'):
    """Load an array (dense or sparse)."""
    with h5py.File(fname, 'r') as f:
        obj = f[key]
        # Inspect the HDF5 object type rather than reading the whole dataset:
        # a SparseArray is stored as a group, a dense array as a dataset.
        if isinstance(obj, h5py.Group):
            if '__bdpy_sparse_arrray' in obj:
                s_ary = SparseArray(fname, key=key)
                return s_ary.dense
            raise RuntimeError('Unsupported group: %s' % key)
        if isinstance(obj, h5py.Dataset):
            # Dense array (read with h5py; hdf5storage breaks under NumPy 2.0)
            return _mat_v73.read_dataset(obj)
        raise RuntimeError('Unsupported data type: %s' % type(obj))


def save_array(fname, array, key='data', dtype=np.float64, sparse=False):
    """Save an array (dense or sparse)."""
    if sparse:
        # Save as a SparseArray
        s_ary = SparseArray(array.astype(dtype))
        s_ary.save(fname, key=key, dtype=dtype)
    else:
        # Save as a dense array
        warnings.warn(_MATLAB_WRITE_FUTURE_WARNING, FutureWarning, stacklevel=2)
        hdf5storage.savemat(fname,
                            {key: array.astype(dtype)},
                            format='7.3', oned_as='column',
                            store_python_metadata=True)

    return None


def save_multiarrays(fname, arrays):
    """Save arrays (dense)."""
    warnings.warn(_MATLAB_WRITE_FUTURE_WARNING, FutureWarning, stacklevel=2)
    save_dict = {k: v for k, v in arrays.items()}
    hdf5storage.savemat(fname,
                        save_dict,
                        format='7.3', oned_as='column',
                        store_python_metadata=True)

    return None


class SparseArray(object):
    """Sparse array class."""
    
    def __init__(self, src=None, key='data', background=0):
        self.__background = background

        if type(src) == np.ndarray:
            # Create sparse array from numpy.ndarray
            self.__make_sparse(src)
        elif os.path.isfile(src):
            # Load data from src
            self.__load(src, key=key)
        else:
            raise ValueError('Unsupported input')

    @property
    def dense(self):
        return self.__make_dense()

    def save(self, fname, key='data', dtype=np.float64):
        warnings.warn(_MATLAB_WRITE_FUTURE_WARNING, FutureWarning, stacklevel=2)
        if _NUMPY2:
            # Avoid hdf5storage.savemat here (it can fail when overwriting an
            # existing sparse struct under NumPy 2.x) and write the struct with
            # h5py instead. We replace only ``key`` in the target file, leaving
            # any other variables already stored there untouched.
            self.__save_h5py(fname, key=key, dtype=dtype)
        else:
            payload = {key: {u'__bdpy_sparse_arrray': True,
                             u'index': self.__index,
                             u'value': self.__value.astype(dtype),
                             u'shape': self.__shape,
                             u'background': self.__background}}
            hdf5storage.savemat(fname, payload, format='7.3',
                                oned_as='column', store_python_metadata=True)
        return None

    def __save_h5py(self, fname, key='data', dtype=np.float64):
        # NumPy-2-only writer for the sparse-array struct. The layout mirrors
        # what ``__load`` expects: ``index``/``shape`` as plain matrices (which
        # _mat_v73.read_cell restores row-by-row) and ``value``/``background``
        # as plain datasets. Opening in append mode and deleting only ``key``
        # avoids the unconditional full-file rewrite of the previous fallback.
        index = np.vstack([np.asarray(i, dtype=np.int64).ravel()
                           for i in self.__index])
        mode = 'a' if os.path.exists(fname) else 'w'
        with h5py.File(fname, mode) as f:
            if key in f:
                del f[key]
            g = f.create_group(key)
            g.create_dataset(u'__bdpy_sparse_arrray', data=True)
            g.create_dataset(u'index', data=index)
            g.create_dataset(u'value', data=self.__value.astype(dtype).ravel())
            g.create_dataset(u'shape', data=np.asarray(self.__shape, dtype=np.int64))
            g.create_dataset(u'background', data=np.asarray(self.__background))
        return None

    def __make_sparse(self, array):
        self.__index = np.where(array != self.__background)
        self.__value = array[self.__index]
        self.__shape = array.shape
        return None

    def __make_dense(self):
        dense = np.ones(self.__shape) * self.__background
        dense[self.__index] = self.__value
        return dense

    def __load(self, fname, key='data'):
        # Read with h5py instead of hdf5storage, which breaks under NumPy 2.0.
        # The struct stores ``index``/``shape`` as cell arrays (object refs) when
        # written by bdpy, or as plain matrices when written by other tools.
        with h5py.File(fname, 'r') as f:
            g = f[key]
            self.__index = tuple(
                np.asarray(c).ravel().astype(int)
                for c in _mat_v73.read_cell(f, g['index'])
            )

            value = np.asarray(_mat_v73.read_dataset(g['value'])).ravel()
            self.__value = value

            self.__shape = tuple(
                int(np.asarray(s).ravel()[0])
                for s in _mat_v73.read_cell(f, g['shape'])
            )

            background = np.asarray(_mat_v73.read_dataset(g['background'])).ravel()
            self.__background = background[0] if background.size else 0
        return None
