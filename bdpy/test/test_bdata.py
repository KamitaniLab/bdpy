"""Tests for bdpy.bdata"""

import unittest
import copy

import numpy as np
from numpy.testing import assert_array_equal

import bdpy


class TestBdata(unittest.TestCase):
    """Tests of 'bdata' module"""

    def __init__(self, *args, **kwargs):

        super(TestBdata, self).__init__(*args, **kwargs)

        self.data = bdpy.BData()

        x = np.array([[0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                      [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                      [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                      [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
        g = np.array([1, 2, 3, 4, 5])

        self.data.add(x, 'VoxelData')
        self.data.add(g, 'Group')

        self.data.add_metadata('Mask_0:3', [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], where='VoxelData')
        self.data.add_metadata('Mask_3:3', [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], where='VoxelData')
        self.data.add_metadata('Mask_6:3', [0, 0, 0, 0, 0, 0, 1, 1, 1, 0], where='VoxelData')
        self.data.add_metadata('Mask_0:5', [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], where='VoxelData')
        self.data.add_metadata('Val_A',    [9, 7, 5, 3, 1, 0, 2, 4, 6, 8], where='VoxelData')

    def test_add(self):
        """Test for add."""

        colname = 'TestData'
        data = np.random.rand(5, 10)

        b = bdpy.BData()
        b.add(data, colname)

        np.testing.assert_array_equal(b.dataSet, data)
        np.testing.assert_array_equal(b.metadata.get(colname, 'value'),
                                      np.array([1] * 10))

    def test_add_2(self):
        """Test for add."""

        colnames = ['TestData1', 'TestData2']
        datalist = [np.random.rand(5, 10),
                    np.random.rand(5, 3)]
        
        b = bdpy.BData()

        for c, d in zip(colnames, datalist):
            b.add(d, c)

        # Test
        np.testing.assert_array_equal(b.dataSet, np.hstack(datalist))

        np.testing.assert_array_equal(b.metadata.get(colnames[0], 'value'),
                                      np.array([1] * 10 + [np.nan] * 3))
        np.testing.assert_array_equal(b.metadata.get(colnames[1], 'value'),
                                      np.array([np.nan] * 10 + [1] * 3))

    def test_add_3(self):
        """Test for add."""

        b = bdpy.BData()
        data_a1 = np.random.rand(10, 10)
        data_b = np.random.rand(10, 3)
        data_a2 = np.random.rand(10, 5)

        b.add(data_a1, 'TestDataA')
        b.add(data_b, 'TestDataB')
        b.add(data_a2, 'TestDataA')

        np.testing.assert_array_equal(b.dataSet, np.hstack((data_a1, data_b, data_a2)))

        np.testing.assert_array_equal(b.metadata.get('TestDataA', 'value'),
                                      np.array([1] * 10 + [np.nan] * 3 + [1] * 5))
        np.testing.assert_array_equal(b.metadata.get('TestDataB', 'value'),
                                      np.array([np.nan] * 10 + [1] * 3 + [np.nan] * 5))

    def test_add_metadata_1(self):
        """Test for add_metadata."""

        md_key = 'TestMetaData'
        md_desc = 'Metadata for test'
        md_val = np.random.rand(10)

        testdata = np.random.rand(10, 10)
        
        b = bdpy.BData()
        b.add(testdata, 'TestData')
        b.add_metadata(md_key, md_val, md_desc)

        assert_array_equal(b.dataSet, testdata)
        assert_array_equal(b.metadata.get('TestData', 'value'),
                           np.array([1] * 10))
        assert_array_equal(b.metadata.get('TestMetaData', 'value'),
                           md_val)

    def test_add_metadata_2(self):
        """Test for add_metadata."""

        md_key_1 = 'TestMetaData1'
        md_desc_1 = 'Metadata for test 1'
        md_val_1 = np.random.rand(10)

        md_key_2 = 'TestMetaData2'
        md_desc_2 = 'Metadata for test 2'
        md_val_2 = np.random.rand(10)

        testdata = np.random.rand(10, 10)
        
        b = bdpy.BData()
        b.add(testdata, 'TestData')
        b.add_metadata(md_key_1, md_val_1, md_desc_1)
        b.add_metadata(md_key_2, md_val_2, md_desc_2)

        assert_array_equal(b.dataSet, testdata)
        assert_array_equal(b.metadata.get('TestData', 'value'),
                           np.array([1] * 10))
        assert_array_equal(b.metadata.get('TestMetaData1', 'value'),
                           md_val_1)
        assert_array_equal(b.metadata.get('TestMetaData2', 'value'),
                           md_val_2)

    def test_add_metadata_3(self):
        """Test for add_metadata."""

        md_key = 'TestMetaData'
        md_desc = 'Metadata for test'
        md_val = np.random.rand(10)

        testdata_a = np.random.rand(10, 10)
        testdata_b = np.random.rand(10, 10)
        
        b = bdpy.BData()
        b.add(testdata_a, 'TestDataA')
        b.add(testdata_b, 'TestDataB')

        b.add_metadata(md_key, md_val, attribute='TestDataA')

        assert_array_equal(b.dataSet, np.hstack((testdata_a, testdata_b)))

        assert_array_equal(b.metadata.get('TestDataA', 'value'),
                           np.array([1] * 10 + [np.nan] * 10))
        assert_array_equal(b.metadata.get('TestMetaData', 'value'),
                           np.hstack((md_val, [np.nan] * 10)))

    def test_add_metadata_4(self):
        """Test for add_metadata."""

        md_key = 'TestMetaData'
        md_desc = 'Metadata for test'
        md_val = np.random.rand(10)

        testdata_a = np.random.rand(10, 10)
        testdata_b = np.random.rand(10, 10)
        
        b = bdpy.BData()
        b.add(testdata_a, 'TestDataA')
        b.add(testdata_b, 'TestDataB')

        b.add_metadata(md_key, md_val, where='TestDataA')

        assert_array_equal(b.dataSet, np.hstack((testdata_a, testdata_b)))

        assert_array_equal(b.metadata.get('TestDataA', 'value'),
                           np.array([1] * 10 + [np.nan] * 10))
        assert_array_equal(b.metadata.get('TestMetaData', 'value'),
                           np.hstack((md_val, [np.nan] * 10)))

    def test_add_metadata_5(self):
        """Test for add_metadata."""

        md_key = 'TestMetaData'
        md_desc = 'Metadata for test'
        md_val = np.random.rand(10)

        testdata_a = np.random.rand(10, 10)
        testdata_b = np.random.rand(10, 10)
        
        b = bdpy.BData()
        b.add(testdata_a, 'TestDataA')
        b.add(testdata_b, 'TestDataB')

        b.add_metadata(md_key, md_val, where='TestDataA', attribute='TestDataB')

        assert_array_equal(b.dataSet, np.hstack((testdata_a, testdata_b)))

        assert_array_equal(b.metadata.get('TestDataA', 'value'),
                           np.array([1] * 10 + [np.nan] * 10))
        assert_array_equal(b.metadata.get('TestMetaData', 'value'),
                           np.hstack((md_val, [np.nan] * 10)))

    def test_set_metadatadescription_1(self):
        '''Test for set_metadatadescription.'''

        expected = 'Test for set_metadatadescription'

        data = copy.deepcopy(self.data)
        data.set_metadatadescription('Val_A', expected)
        actual = data.metadata.description[-1]

        self.assertEqual(actual, expected)

    def test_get_1(self):
        """Test for get."""

        b = bdpy.BData()

        attr_A = 'TestAttr_A'
        data_A = np.random.rand(10, 10)
        b.add(data_A, attr_A)

        attr_B = 'TestAttr_B'
        data_B = np.random.rand(10, 2)
        b.add(data_B, attr_B)

        exp_output = np.hstack((data_A, data_B))
        test_output = b.get()

        np.testing.assert_array_equal(test_output, exp_output)

    def test_get_2(self):
        """Test for get."""

        b = bdpy.BData()

        attr_A = 'TestAttr_A'
        data_A = np.random.rand(10, 10)
        b.add(data_A, attr_A)

        attr_B = 'TestAttr_B'
        data_B = np.random.rand(10, 2)
        b.add(data_B, attr_B)

        exp_output = data_A
        test_output = b.get('TestAttr_A')

        np.testing.assert_array_equal(test_output, exp_output)

    ## Tests for select() ##########

    def test_select_pass0001(self):
        """Test for '='"""

        test_input = 'Mask_0:3 = 1'
        exp_output = self.data.dataSet[:, 0:3]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0002(self):
        """Test for '|' (or)"""

        test_input = 'Mask_0:3 = 1 | Mask_3:3 = 1'
        exp_output = self.data.dataSet[:, 0:6]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0003(self):
        """Test for '&' (and)"""

        test_input = 'Mask_0:3 = 1 & Mask_0:5 = 1'
        exp_output = self.data.dataSet[:, 0:3]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0004(self):
        """Test for three condition terms"""

        test_input = 'Mask_0:5 = 1 & Mask_0:3 = 1 | Mask_3:3 = 1'
        exp_output = self.data.dataSet[:, 0:6]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0005(self):
        """Test for parentheses"""

        test_input = 'Mask_0:5 = 1 & (Mask_0:3 = 1 | Mask_3:3 = 1)'
        exp_output = self.data.dataSet[:, 0:5]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    # def test_select_pass0006(self):
    #     """Test for 'top'"""

    #     test_input = 'Val_A top 5'
    #     exp_output = self.data.dataSet[:, np.array([0, 1, 2, 8, 9], dtype=int)]

    #     test_output = self.data.select(test_input)

    #     np.testing.assert_array_equal(test_output, exp_output)

    # def test_select_pass0007(self):
    #     """Test for 'top'"""

    #     test_input = 'Val_A top 10'
    #     exp_output = self.data.dataSet[:, 0:10]

    #     test_output = self.data.select(test_input)

    #     np.testing.assert_array_equal(test_output, exp_output)

    # def test_select_pass0008(self):
    #     """Test for 'top' and '@'"""

    #     test_input = 'Val_A top 3 @ Mask_0:5 = 1'
    #     exp_output = self.data.dataSet[:, 0:3]

    #     test_output = self.data.select(test_input)

    #     np.testing.assert_array_equal(test_output, exp_output)

    # def test_select_pass0009(self):
    #     """Test for 'top' and '@'"""

    #     test_input = 'Val_A top 3 @ (Mask_3:3 = 1 | Mask_6:3 = 1)'
    #     exp_output = self.data.dataSet[:, [3, 7, 8]]

    #     test_output = self.data.select(test_input)

    #     np.testing.assert_array_equal(test_output, exp_output)

    # def test_select_pass0010(self):
    #     """Test for 'top' and '@'"""

    #     test_input = 'Val_A top 3 @ Mask_3:3 = 1 | Mask_6:3 = 1'
    #     exp_output = self.data.dataSet[:, [3, 7, 8]]

    #     test_output = self.data.select(test_input)

    #     np.testing.assert_array_equal(test_output, exp_output)

if __name__ == "__main__":
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBdata)
    unittest.TextTestRunner(verbosity=2).run(test_suite)
