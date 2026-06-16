'''Tests for bdpy.bdata.bdata.'''


import os
import tempfile
import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal

from bdpy.bdata.bdata import BData


class TestBdata(unittest.TestCase):
    '''Tests of 'bdata' module'''

    def __init__(self, *args, **kwargs):
        super(TestBdata, self).__init__(*args, **kwargs)

    def test_add_get(self):
        '''Test for BData.add and get.'''
        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)
        data_z = np.random.rand(5, 20)

        b = BData()

        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')
        b.add(data_z, 'Data_Z')

        # dataset
        assert_array_equal(b.get('Data_X'), data_x)
        assert_array_equal(b.get('Data_Y'), data_y)
        assert_array_equal(b.get('Data_Z'), data_z)

        # metadata
        assert_array_equal(b.metadata.get('Data_X', 'value'), np.array([1] * 10 + [np.nan] * 8 + [np.nan] * 20))
        assert_array_equal(b.metadata.get('Data_Y', 'value'), np.array([np.nan] * 10 + [1] * 8 + [np.nan] * 20))
        assert_array_equal(b.metadata.get('Data_Z', 'value'), np.array([np.nan] * 10 + [np.nan] * 8 + [1] * 20))

        # metadata (BData.get_metadata)
        assert_array_equal(b.get_metadata('Data_X'), np.array([1] * 10 + [np.nan] * 8 + [np.nan] * 20))
        assert_array_equal(b.get_metadata('Data_Y'), np.array([np.nan] * 10 + [1] * 8 + [np.nan] * 20))
        assert_array_equal(b.get_metadata('Data_Z'), np.array([np.nan] * 10 + [np.nan] * 8 + [1] * 20))

    def test_add_dataset_is_obsoleted(self):
        '''Test that BData.add_dataset warns once and delegates to add.'''
        b = BData()

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter('always')
            b.add_dataset(np.ones((2, 3)), 'Data')

        self.assertEqual(len(recorded_warnings), 1)
        self.assertIn("'add_dataset' is obsoleted", str(recorded_warnings[0].message))
        assert_array_equal(b.get('Data'), np.ones((2, 3)))

    def test_metadata_add_get(self):
        '''Test for add/get_metadata.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)

        n_col = data_x.shape[1] + data_y.shape[1]

        metadata_a = np.random.rand(n_col)
        metadata_b = np.random.rand(n_col)

        b = BData()

        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        b.add_metadata('Metadata_A', metadata_a)
        b.add_metadata('Metadata_B', metadata_b)

        assert_array_equal(b.metadata.get('Metadata_A', 'value'), metadata_a)
        assert_array_equal(b.metadata.get('Metadata_B', 'value'), metadata_b)
        assert_array_equal(b.get_metadata('Metadata_A'), metadata_a)
        assert_array_equal(b.get_metadata('Metadata_B'), metadata_b)

    def test_metadata_add_get_where(self):
        '''Test for add/get_metadata with where option.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)

        metadata_a = np.random.rand(10)
        metadata_b = np.random.rand(8)

        b = BData()

        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        b.add_metadata('Metadata_A', metadata_a, where='Data_X')
        b.add_metadata('Metadata_B', metadata_b, where='Data_Y')

        assert_array_equal(b.get_metadata('Metadata_A'), np.hstack([metadata_a, np.array([np.nan] * 8)]))
        assert_array_equal(b.get_metadata('Metadata_B'), np.hstack([np.array([np.nan] * 10), metadata_b]))
        assert_array_equal(b.get_metadata('Metadata_A', where='Data_X'), metadata_a)
        assert_array_equal(b.get_metadata('Metadata_B', where='Data_Y'), metadata_b)

    def test_get_metadata_notfound(self):
        '''Test for BData.get_metadata with a missing key.'''
        b = BData()
        b.add(np.ones((2, 3)), 'Data')

        self.assertIsNone(b.get_metadata('Metadata_NotFound'))

    def test_get_metadata_notfound_where(self):
        '''Test for BData.get_metadata with a missing key and where option.'''
        b = BData()
        b.add(np.ones((2, 3)), 'Data')

        with self.assertRaises(ValueError):
            b.get_metadata('Metadata_NotFound', where='Data')

    def test_update_notfound(self):
        '''Test for BData.update with a missing key.'''
        b = BData()
        b.add(np.ones((2, 3)), 'Data')

        with self.assertRaises(ValueError):
            b.update('Metadata_NotFound', np.zeros((2, 3)))

    def test_set_metadatadescription_1(self):
        '''Test for set_metadatadescription.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)

        metadata_a = np.random.rand(10)
        metadata_b = np.random.rand(8)

        b = BData()
        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')
        b.add_metadata('Metadata_A', metadata_a, where='Data_X')
        b.add_metadata('Metadata_B', metadata_b, where='Data_Y')

        metadata_desc = 'Test metadata description'

        b.set_metadatadescription('Metadata_A', metadata_desc)

        self.assertEqual(b.metadata.get('Metadata_A', 'description'), metadata_desc)

    def test_select(self):
        '''Test for BData.select.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 5)

        b = BData()
        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        b.add_metadata('ROI_0:5', [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], where='Data_X')
        b.add_metadata('ROI_3:8', [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], where='Data_X')
        b.add_metadata('ROI_4:9', [0, 0, 0, 0, 1, 1, 1, 1, 1, 0], where='Data_X')

        assert_array_equal(b.select('Data_X'), data_x)
        assert_array_equal(b.select('Data_X = 1'), data_x)

        assert_array_equal(b.select('ROI_0:5'), data_x[:, 0:5])
        assert_array_equal(b.select('ROI_0:5 & ROI_3:8'), data_x[:, 3:5])
        assert_array_equal(b.select('ROI_0:5 = 1 & ROI_3:8 = 1'), data_x[:, 3:5])
        assert_array_equal(b.select('ROI_0:5 | ROI_3:8'), data_x[:, 0:8])
        assert_array_equal(b.select('ROI_0:5 = 1 | ROI_3:8 = 1'), data_x[:, 0:8])
        assert_array_equal(b.select('(ROI_0:5 | ROI_3:8) & ROI_4:9'), data_x[:, 4:8])
        assert_array_equal(b.select('ROI_0:5 | (ROI_3:8 & ROI_4:9)'), data_x[:, 0:8])

        assert_array_equal(b.select('ROI_*'), data_x[:, 0:9])
        assert_array_equal(b.select('ROI_0:5 + ROI_3:8'), data_x[:, 0:8])
        assert_array_equal(b.select('ROI_0:5 - ROI_3:8'), data_x[:, 0:3])

    def test_select_return_index(self):
        '''Test for BData.select with return_index=True.'''
        data_x = np.arange(20, dtype=float).reshape(4, 5)
        data_y = np.arange(8, dtype=float).reshape(4, 2)

        b = BData()

        # BData.add stacks arrays column-wise: Data_X columns come first,
        # followed by Data_Y columns.
        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        # add_metadata(..., where='Data_X') defines the ROI only inside the
        # Data_X column group, not against the whole dataset.
        b.add_metadata('ROI_1:4', [0, 1, 1, 1, 0], where='Data_X')

        selected_data, selected_index = b.select('ROI_1:4', return_index=True)

        expected_data_x_index = np.array([False, True, True, True, False])
        expected_data_y_index = np.array([False, False])
        expected_dataset_index = np.hstack([expected_data_x_index, expected_data_y_index])

        assert_array_equal(selected_data, data_x[:, 1:4])
        self.assertEqual(len(selected_index), b.dataset.shape[1])
        assert_array_equal(selected_index, expected_dataset_index)
        self.assertEqual(selected_index.dtype, np.dtype(bool))

    def test_applyfunc_with_selected_columns(self):
        '''Test for BData.applyfunc with a selected column group.'''
        data_x = np.arange(6, dtype=float).reshape(3, 2)
        data_y = np.arange(3, dtype=float).reshape(3, 1)

        b = BData()
        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        b.applyfunc(lambda x: x + 10, where='Data_X')

        assert_array_equal(b.get('Data_X'), data_x + 10)
        assert_array_equal(b.get('Data_Y'), data_y)

    def test_applyfunc_tuple_result_reindexes_all_columns(self):
        '''Test that BData.applyfunc reindexes all columns when func returns an index map.'''
        data_x = np.arange(6, dtype=float).reshape(3, 2)
        data_y = np.arange(3, dtype=float).reshape(3, 1)
        row_index = np.array([2, 0])

        b = BData()
        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        # A tuple result means that the selected column group is replaced by
        # the first element, and every non-selected column follows the returned
        # row index map so that rows remain aligned across the whole dataset.
        b.applyfunc(lambda x: (x[row_index], row_index), where='Data_X')

        assert_array_equal(b.get('Data_X'), data_x[row_index])
        assert_array_equal(b.get('Data_Y'), data_y[row_index])

    def test_save_load_hdf5_header_and_vmap(self):
        '''Test for HDF5 roundtrip of header values and vmap.'''
        data = np.arange(6, dtype=float).reshape(3, 2)
        label = np.array([1, 2, 1], dtype=float).reshape(3, 1)

        bdata = BData()
        bdata.add(data, 'Data')
        bdata.add(label, 'Label')
        bdata.add_vmap('Label', {1: 'label-1', 2: 'label-2'})
        bdata.update_header({
            'source': 'manual',
            'indices': [1, 2],
            'scale': 1.5,
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, 'test_bdata.h5')
            bdata.save(h5_path, 'HDF5')
            loaded_bdata = BData(h5_path, 'HDF5')

        assert_array_equal(loaded_bdata.get('Data'), data)
        assert_array_equal(loaded_bdata.get('Label'), label)
        self.assertEqual(loaded_bdata.get_vmap('Label'), {1.0: 'label-1', 2.0: 'label-2'})
        self.assertEqual(loaded_bdata.header['source'], 'manual')
        self.assertEqual(loaded_bdata.header['indices'], [1, 2])
        self.assertEqual(loaded_bdata.header['scale'], 1.5)

    # Tests for vmap
    def test_vmap_add_get(self):
        bdata = BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4'}
        label = ['label-1', 'label-2', 'label-3', 'label-4']

        bdata.add_vmap('Label', label_map)
        assert bdata.get_vmap('Label') == label_map
        self.assertEqual(set(bdata.get_vmap_keys()), {'Label'})

        # Get labels
        np.testing.assert_array_equal(bdata.get_label('Label'), label)

    def test_vmap_add_same_map(self):
        bdata = BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4'}
        label = ['label-1', 'label-2', 'label-3', 'label-4']

        bdata.add_vmap('Label', label_map)
        bdata.add_vmap('Label', label_map)
        assert bdata.get_vmap('Label') == label_map

        # Get labels
        np.testing.assert_array_equal(bdata.get_label('Label'), label)

    def test_vmap_errorcases(self):
        n_sample = 4

        bdata = BData()
        bdata.add(np.random.rand(n_sample, 3), 'MainData')
        bdata.add(np.arange(n_sample) + 1, 'Label')

        label_map = {(i + 1): 'label-%04d' % (i + 1) for i in range(n_sample)}

        bdata.add_vmap('Label', label_map)

        # Vmap not found
        with self.assertRaises(ValueError):
            bdata.get_label('MainData')

        # Invalid vmap (map is not a dict)
        label_map_invalid = range(n_sample)
        with self.assertRaises(TypeError):
            bdata.add_vmap('Label', label_map_invalid)

        # Invalid vmap (key is str)
        label_map_invalid = {'label-%04d' % i: i for i in range(n_sample)}
        with self.assertRaises(TypeError):
            bdata.add_vmap('Label', label_map_invalid)

        # Inconsistent vmap
        label_map_inconsist = {i: 'label-%04d-inconsist' % i
                               for i in range(n_sample)}
        with self.assertRaises(ValueError):
            bdata.add_vmap('Label', label_map_inconsist)

    def test_vmap_add_unnecessary_vmap(self):
        bdata = BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4',
                     5: 'label-5'}
        label_map_ture = {1: 'label-1',
                          2: 'label-2',
                          3: 'label-3',
                          4: 'label-4'}

        bdata.add_vmap('Label', label_map)
        assert bdata.get_vmap('Label') == label_map_ture

    def test_vmap_add_insufficient_vmap(self):
        bdata = BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3'}

        with self.assertRaises(ValueError):
            bdata.add_vmap('Label', label_map)

    def test_vmap_add_invalid_name_vmap(self):
        bdata = BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4'}

        with self.assertRaises(ValueError):
            bdata.add_vmap('InvalidLabel', label_map)


if __name__ == "__main__":
    unittest.main()
