'''Tests for dataform'''

import os
import tempfile
import unittest

import h5py
import numpy as np

from bdpy.dataform.sparse import load_array, save_array


class TestSparse(unittest.TestCase):

    def test_load_save_dense_array(self):
        payloads = [
            [(10,), 'test_array_dense_ndim1.mat'],  # ndim = 1
            [(3, 2), 'test_array_dense_ndim2.mat'],  # ndim = 2
            [(4, 3, 2), 'test_array_dense_ndim3.mat']  # ndim = 3
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for shape, fname in payloads:
                original_data = np.random.rand(*shape)
                save_array(tmpdir + '/' + fname, original_data, key='testdata')
                from_file = load_array(tmpdir + '/' + fname, key='testdata')

                np.testing.assert_array_equal(original_data, from_file)

    def test_load_save_sparse_array(self):
        payloads = [
            [(10,), 'test_array_sparse_ndim1.mat'],  # ndim = 1
            [(3, 2), 'test_array_sparse_ndim2.mat'],  # ndim = 2
            [(4, 3, 2), 'test_array_sparse_ndim3.mat']  # ndim = 3
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for shape, fname in payloads:
                original_data = np.random.rand(*shape)
                original_data[original_data < 0.8] = 0

                save_array(tmpdir + '/' + fname, original_data, key='testdata', sparse=True)
                from_file = load_array(tmpdir + '/' + fname, key='testdata')

                np.testing.assert_array_equal(original_data, from_file)

    def test_sparse_save_preserves_other_variables_under_numpy2(self):
        if int(np.__version__.split('.')[0]) < 2:
            self.skipTest('NumPy-2-only h5py writer')

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_sparse_preserve.mat')

            with h5py.File(fname, 'w') as f:
                f.create_dataset('other', data=np.array([1, 2, 3]))

            original_data = np.random.rand(3, 2)
            original_data[original_data < 0.8] = 0

            save_array(fname, original_data, key='data', sparse=True)
            from_file = load_array(fname, key='data')

            np.testing.assert_array_equal(original_data, from_file)

            with h5py.File(fname, 'r') as f:
                np.testing.assert_array_equal(f['other'][()], np.array([1, 2, 3]))

    def test_load_array_jl(self):
        data = np.array([[1, 0, 0, 0],
                         [2, 2, 0, 0],
                         [3, 3, 3, 0]])
        data_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'data'
        ))

        testdata = load_array(
            os.path.join(data_dir, 'array_jl_dense_v1.mat'), key='a')
        np.testing.assert_array_equal(data, testdata)

        testdata = load_array(
            os.path.join(data_dir, 'array_jl_sparse_v1.mat'), key='a')
        np.testing.assert_array_equal(data, testdata)


if __name__ == '__main__':
    unittest.main()
