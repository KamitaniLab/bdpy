"""Read MATLAB v7.3 (HDF5) ``.mat`` files with h5py.

bdpy historically relied on :func:`hdf5storage.loadmat` for the load path, but
that broke under NumPy 2.0, which removed ``np.unicode_`` that older hdf5storage
referenced (see issue #106). This module reimplements the *read* side on top of
h5py for the data layouts bdpy actually uses (dense numeric arrays and the
sparse-array struct). Saving still goes through hdf5storage.

This file is a part of BdPy.
"""

import h5py
import numpy as np
import scipy.io as sio

__all__ = ["load_array", "loadmat_key", "read_cell", "read_dataset"]


def read_dataset(dset: h5py.Dataset) -> np.ndarray:
    """Read an h5py dataset, undoing MATLAB v7.3 / hdf5storage conventions.

    MATLAB stores arrays in Fortran (column-major) order, so multi-dimensional
    datasets are written transposed relative to NumPy's C order. hdf5storage
    additionally records the original Python shape and empty-array flags as
    ``Python.*`` attributes, which we honor to reproduce ``hdf5storage.loadmat``.

    Parameters
    ----------
    dset : h5py.Dataset
        Dataset to read.

    Returns
    -------
    numpy.ndarray
        The array with its original shape restored.
    """
    attrs = dset.attrs
    if "Python.Empty" in attrs or "MATLAB_empty" in attrs:
        shape = tuple(int(x) for x in attrs.get("Python.Shape", ()))
        return np.empty(shape, dtype=dset.dtype)
    arr = dset[()]
    if "MATLAB_class" in attrs and isinstance(arr, np.ndarray) and arr.ndim >= 2:
        arr = np.transpose(arr)
    if "Python.Shape" in attrs:
        arr = np.asarray(arr).reshape(tuple(int(x) for x in attrs["Python.Shape"]))
    return np.asarray(arr)


def read_cell(f: h5py.File, dset: h5py.Dataset) -> list:
    """Read a MATLAB cell array (or a plain matrix) into a list of arrays.

    hdf5storage stores Python tuples/lists as MATLAB cell arrays, i.e. an object
    dataset of HDF5 references to the individual elements. Files written by other
    tools (e.g. MATLAB or Julia) may instead store the same information as a plain
    2-D matrix whose rows are the elements; both layouts are handled here.

    Parameters
    ----------
    f : h5py.File
        Open file, used to dereference cell-array element references.
    dset : h5py.Dataset
        The cell (object) dataset or a plain matrix.

    Returns
    -------
    list of numpy.ndarray
        One array per cell element / matrix row.
    """
    data = dset[()]
    if isinstance(data, np.ndarray) and data.dtype == object:
        return [read_dataset(f[ref]) for ref in data.ravel()]
    arr = np.asarray(data)
    return [arr[i] for i in range(arr.shape[0])]


def load_array(path: str, key: str) -> np.ndarray:
    """Load a single dense numeric array from a v7.3 ``.mat`` file.

    Parameters
    ----------
    path : str
        Path to the ``.mat`` file.
    key : str
        Variable name to load.

    Returns
    -------
    numpy.ndarray
        The loaded array.
    """
    with h5py.File(path, "r") as f:
        return read_dataset(f[key])


def loadmat_key(path: str, key: str) -> np.ndarray:
    """Load one variable from a ``.mat`` file, handling both v5 and v7.3.

    MATLAB v5 (and earlier) files are read with :func:`scipy.io.loadmat`; v7.3
    (HDF5) files, which scipy cannot read, fall back to the h5py reader. This
    preserves v5 support while avoiding hdf5storage on the load path (which
    breaks under NumPy 2.0).

    Parameters
    ----------
    path : str
        Path to the ``.mat`` file.
    key : str
        Variable name to load.

    Returns
    -------
    numpy.ndarray
        The loaded array.
    """
    try:
        array = sio.loadmat(path)[key]
    except (NotImplementedError, ValueError):
        return load_array(path, key)
    return np.asarray(array)
