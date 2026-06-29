"""h5py-based legacy reading support for MATLAB v7.3 / hdf5storage ``.mat`` files.

bdpy historically relied on a third-party MATLAB-v7.3 library (hdf5storage) to
read ``.mat`` files, but it broke under NumPy 2.0 (it referenced the removed
``np.unicode_``; see issue #106) and added an extra dependency. This module
replaces that *read* path with h5py, understanding the MATLAB v7.3 / hdf5storage
on-disk conventions (column-major / reversed dimension order, ``MATLAB_class``
and ``Python.*`` attributes), so existing files still load correctly.

This module is **read-only**. bdpy no longer writes MATLAB-compatible ``.mat``
files; new files are saved as bdpy-native plain HDF5 directly at the save sites
(see ``bdpy/dataform/sparse.py`` and ``bdpy/dataform/features.py``). The only
compatibility promise is reading existing hdf5storage / MATLAB v7.3 / bdpy files.

This file is a part of BdPy.
"""

import h5py
import numpy as np
import scipy.io as sio

__all__ = [
    "load_array",
    "loadmat_key",
    "read_cell",
    "read_dataset",
]


def read_dataset(dset: h5py.Dataset) -> np.ndarray:
    """Read an h5py dataset, undoing MATLAB v7.3 conventions.

    MATLAB stores arrays in Fortran (column-major) order, so multi-dimensional
    datasets are written transposed relative to NumPy's C order. Older bdpy
    files additionally record the original Python shape and empty-array flags as
    ``Python.*`` attributes, which we honor to reproduce the original array.
    Only ``Python.Empty`` is special-cased; a bare ``MATLAB_empty`` (set by
    MATLAB without ``Python.Shape``) is read through the normal path so that
    empty non-scalar arrays such as ``(0, 3)`` keep their shape.

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
    if "Python.Empty" in attrs:
        # Only Python.Empty (written by the legacy writer) implies a Python.Shape
        # we can trust; fall back to the stored dataset shape if it is missing. A
        # bare MATLAB_empty (written by MATLAB without Python.Shape) must NOT be
        # treated this way -- np.empty(()) would collapse e.g. (0, 3) to 0-d --
        # so it falls through to the normal read/transpose path below.
        shape = tuple(int(x) for x in attrs.get("Python.Shape", dset.shape))
        return np.empty(shape, dtype=dset.dtype)
    arr = dset[()]
    if "MATLAB_class" in attrs and isinstance(arr, np.ndarray) and arr.ndim >= 2:
        arr = np.transpose(arr)
    if "Python.Shape" in attrs:
        arr = np.asarray(arr).reshape(tuple(int(x) for x in attrs["Python.Shape"]))
    return np.asarray(arr)


def read_cell(f: h5py.File, dset: h5py.Dataset) -> list:
    """Read a MATLAB cell array (or a plain matrix) into a list of arrays.

    Python tuples/lists are stored as MATLAB cell arrays, i.e. an object
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
    # Plain-matrix layout: route through read_dataset so MATLAB-style transposed
    # matrices carrying MATLAB_class are de-transposed before we split rows.
    arr = np.asarray(read_dataset(dset))
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
    preserves v5 support while avoiding the legacy MATLAB-v7.3 library on the
    load path (which breaks under NumPy 2.0).

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
