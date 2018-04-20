"""Tests for bdpy.bdata"""

import unittest
import numpy as np

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

        self.data.add_metadata('Mask_0:3', [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], attribute='VoxelData')
        self.data.add_metadata('Mask_3:3', [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], attribute='VoxelData')
        self.data.add_metadata('Mask_6:3', [0, 0, 0, 0, 0, 0, 1, 1, 1, 0], attribute='VoxelData')
        self.data.add_metadata('Mask_0:5', [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], attribute='VoxelData')
        self.data.add_metadata('Val_A',    [9, 7, 5, 3, 1, 0, 2, 4, 6, 8], attribute='VoxelData')

    def test_add_pass0001(self):
        """Test for add (pass case 0001)"""

        b = bdpy.BData()

        attr = 'TestAttr'
        data = np.random.rand(5, 10)

        b.add(data, attr)

        np.testing.assert_array_equal(b.dataSet, data)
        self.assertEqual(b.metaData[0]['key'], attr)
        #self.assertEqual(b.metaData[0]['description'], 'Attribute: TestAttr = 1')
        np.testing.assert_array_equal(b.metaData[0]['value'], np.ones(10))

    def test_add_pass0002(self):
        """Test for add (pass case 0002)"""

        b = bdpy.BData()

        attr_A = 'TestAttr_A'
        data_A = np.random.rand(10, 10)
        b.add(data_A, attr_A)

        attr_B = 'TestAttr_B'
        data_B = np.random.rand(10, 2)
        b.add(data_B, attr_B)

        np.testing.assert_array_equal(b.dataSet, np.hstack((data_A, data_B)))

        self.assertEqual(b.metaData[0]['key'], attr_A)
        #self.assertEqual(b.metaData[0]['description'], 'Attribute: TestAttr_A = 1')
        np.testing.assert_array_equal(b.metaData[0]['value'], np.hstack((np.ones(10, dtype=int), [np.nan, np.nan])))

        self.assertEqual(b.metaData[1]['key'], attr_B)
        #self.assertEqual(b.metaData[1]['description'], 'Attribute: TestAttr_B = 1')
        np.testing.assert_array_equal(b.metaData[1]['value'], np.hstack(([np.nan for _ in xrange(10)], np.ones(2, dtype=int))))

    def test_add_pass0003(self):
        """Test for add (pass case 0003)"""

        b = bdpy.BData()

        attr_A = 'TestAttr_A'
        data_A = np.random.rand(10, 10)
        b.add(data_A, attr_A)

        attr_B = 'TestAttr_B'
        data_B = np.random.rand(10, 2)
        b.add(data_B, attr_B)

        attr_A2 = 'TestAttr_A'
        data_A2 = np.random.rand(10, 3)
        b.add(data_A2, attr_A2)

        np.testing.assert_array_equal(b.dataSet, np.hstack((data_A, data_B, data_A2)))

        self.assertEqual(b.metaData[0]['key'], attr_A)
        #self.assertEqual(b.metaData[0]['description'], 'Attribute: TestAttr_A = 1')
        np.testing.assert_array_equal(b.metaData[0]['value'],
                                      np.hstack((np.ones(10, dtype=int),
                                                 [np.nan, np.nan],
                                                 np.ones(3, dtype=int))))

    def test_add_metadata_pass0001(self):
        """Test for add_metadata (pass case 0001)"""

        b = bdpy.BData()

        md_key = 'MetaData_A'
        md_desc = 'Metadata for test'
        md_val = np.zeros(10)

        b.add(np.random.rand(10, 10), 'Test data')
        b.add_metadata(md_key, md_val, md_desc)

        exp_metaData = [{'key': 'Test data',
                         'description': '',
                         'value': np.ones(10)},
                        {'key': md_key,
                         'description': md_desc,
                         'value': md_val}]

        for m, e in zip(b.metaData, exp_metaData):
            self.assertEqual(m['key'], e['key'])
            #self.assertEqual(m['description'], e['description'])
            np.testing.assert_array_equal(m['value'], e['value'])

    def test_add_metadata_pass0002(self):
        """Test for add_metadata (pass case 0001)"""

        b = bdpy.BData()

        md_key = 'MetaData_A'

        md_desc_1 = 'Metadata for test'
        md_val_1  = np.zeros(10)

        md_desc_2 = 'Metadata for test (overwriting)'
        md_val_2  = np.ones(10)

        b.add(np.random.rand(10, 10), 'Test data')
        b.add_metadata(md_key, md_val_1, md_desc_1)
        b.add_metadata(md_key, md_val_2, md_desc_2)

        exp_metaData = [{'key': 'Test data',
                         'description': '',
                         'value': np.ones(10)},
                        {'key': md_key,
                         'description': md_desc_2,
                         'value': md_val_2}]

        for m, e in zip(b.metaData, exp_metaData):
            self.assertEqual(m['key'], e['key'])
            #self.assertEqual(m['description'], e['description'])
            np.testing.assert_array_equal(m['value'], e['value'])

    def test_get_pass0001(self):
        """Test for get (pass case 0001)"""

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

    def test_get_pass0002(self):
        """Test for get (pass case 0002)"""

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

    def test_select_pass0006(self):
        """Test for 'top'"""

        test_input = 'Val_A top 5'
        exp_output = self.data.dataSet[:, np.array([0, 1, 2, 8, 9], dtype=int)]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0007(self):
        """Test for 'top'"""

        test_input = 'Val_A top 10'
        exp_output = self.data.dataSet[:, 0:10]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0008(self):
        """Test for 'top' and '@'"""

        test_input = 'Val_A top 3 @ Mask_0:5 = 1'
        exp_output = self.data.dataSet[:, 0:3]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0009(self):
        """Test for 'top' and '@'"""

        test_input = 'Val_A top 3 @ (Mask_3:3 = 1 | Mask_6:3 = 1)'
        exp_output = self.data.dataSet[:, [3, 7, 8]]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_select_pass0010(self):
        """Test for 'top' and '@'"""

        test_input = 'Val_A top 3 @ Mask_3:3 = 1 | Mask_6:3 = 1'
        exp_output = self.data.dataSet[:, [3, 7, 8]]

        test_output = self.data.select(test_input)

        np.testing.assert_array_equal(test_output, exp_output)

if __name__ == "__main__":
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBdata)
    unittest.TextTestRunner(verbosity=2).run(test_suite)
