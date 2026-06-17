"""Tests for bdpy.dataform._mat_v73."""

import os
import tempfile
import unittest

import h5py
import numpy as np

from bdpy.dataform import _mat_v73


class TestReadDataset(unittest.TestCase):

    def test_matlab_empty_without_python_shape_preserves_shape(self):
        # MATLAB-written v7.3 files mark empty arrays with ``MATLAB_empty`` but
        # do not store ``Python.Shape``. Such a dataset must not be collapsed to
        # a 0-d array (the bug fixed here treated MATLAB_empty like Python.Empty
        # and returned ``np.empty(())``).
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'matlab_empty.mat')

            # On-disk shape (0, 3); with MATLAB_class the read path transposes
            # it to (3, 0) like any other MATLAB matrix.
            with h5py.File(fname, 'w') as f:
                dset = f.create_dataset('a', shape=(0, 3), dtype='float64')
                dset.attrs['MATLAB_empty'] = np.uint8(1)
                dset.attrs['MATLAB_class'] = np.bytes_(b'double')
                self.assertNotIn('Python.Shape', dset.attrs)

            with h5py.File(fname, 'r') as f:
                out = _mat_v73.read_dataset(f['a'])

            self.assertNotEqual(out.shape, ())  # would fail with the old code
            self.assertEqual(out.ndim, 2)
            self.assertEqual(out.size, 0)
            self.assertEqual(out.shape, (3, 0))

    def test_matlab_empty_without_matlab_class_keeps_on_disk_shape(self):
        # Without MATLAB_class there is no transpose, so the empty non-scalar
        # shape is returned as stored.
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'matlab_empty_no_class.mat')

            with h5py.File(fname, 'w') as f:
                dset = f.create_dataset('a', shape=(3, 0), dtype='float64')
                dset.attrs['MATLAB_empty'] = np.uint8(1)
                self.assertNotIn('Python.Shape', dset.attrs)

            with h5py.File(fname, 'r') as f:
                out = _mat_v73.read_dataset(f['a'])

            self.assertNotEqual(out.shape, ())
            self.assertEqual(out.shape, (3, 0))
            self.assertEqual(out.size, 0)


if __name__ == '__main__':
    unittest.main()
