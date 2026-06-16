"""Sparse array class.

This file is a part of bdpy.
"""

__all__ = ['SparseArray', 'load_array', 'save_array', 'save_multiarrays']


import os

import h5py
import hdf5storage
import numpy as np

from . import _mat_v73

# hdf5storage.savemat can fail for SparseArray.save under NumPy 2.x, especially
# when overwriting an existing sparse struct. SparseArray.save therefore uses a
# direct h5py writer only under NumPy >= 2. Dense-array save paths are unchanged
# and still rely on hdf5storage for MATLAB-v7.3 metadata/compatibility behavior.
_NUMPY2 = int(np.__version__.split('.')[0]) >= 2


def load_array(fname, key='data'):
    """Load an array (dense or sparse)."""
    with h5py.File(fname, 'r') as f:
        methods = [attr for attr in dir(f[key]) if callable(getattr(f[key], str(attr)))]
        if 'keys' in methods and '__bdpy_sparse_arrray' in f[key].keys():
            # SparseArray
            s_ary = SparseArray(fname, key=key)
            return s_ary.dense
        elif type(f[key][()]) == np.ndarray:
            # Dense array (read with h5py; hdf5storage breaks under NumPy 2.0)
            return _mat_v73.read_dataset(f[key])
        else:
            raise RuntimeError('Unsupported data type: %s' % type(f[key][()]))


def save_array(fname, array, key='data', dtype=np.float64, sparse=False):
    """Save an array (dense or sparse)."""
    if sparse:
        # Save as a SparseArray
        s_ary = SparseArray(array.astype(dtype))
        s_ary.save(fname, key=key, dtype=dtype)
    else:
        # Save as a dense array
        hdf5storage.savemat(fname,
                            {key: array.astype(dtype)},
                            format='7.3', oned_as='column',
                            store_python_metadata=True)

    return None


def save_multiarrays(fname, arrays):
    """Save arrays (dense)."""
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
