'''Tests for bdpy.bdata.metadata.'''


import unittest

import numpy as np
from numpy.testing import assert_array_equal

from bdpy.bdata import metadata


class TestMetadata(unittest.TestCase):
    '''Tests for bdpy.bdata.metadata.'''

    def __init__(self, *args, **kwargs):
        super(TestMetadata, self).__init__(*args, **kwargs)

    def test_set_get(self):
        '''Test for MetaData.set() and MetaData.get().'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1] * 10 + [0] * 5, 'Test metadata A')
        md.set('MetaData_B', [0] * 10 + [1] * 5, 'Test metadata B')

        assert_array_equal(md.get('MetaData_A', 'value'), [1] * 10 + [0] * 5)
        assert_array_equal(md.get('MetaData_A', 'description'), 'Test metadata A')
        assert_array_equal(md.get('MetaData_B', 'value'), [0] * 10 + [1] * 5)
        assert_array_equal(md.get('MetaData_B', 'description'), 'Test metadata B')

    def test_set_get_resize(self):
        '''Test for MetaData.set() and MetaData.get(); resizing values.'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1] * 10 + [0] * 5, 'Test metadata A')
        md.set('MetaData_B', [0] * 10 + [1] * 5, 'Test metadata B')
        md.set('MetaData_C', [0] * 15 + [1] * 3, 'Test metadata C')

        assert_array_equal(md.get('MetaData_A', 'value'), [1] * 10 + [0] * 5 + [np.nan] * 3)
        assert_array_equal(md.get('MetaData_A', 'description'), 'Test metadata A')
        assert_array_equal(md.get('MetaData_B', 'value'), [0] * 10 + [1] * 5 + [np.nan] * 3)
        assert_array_equal(md.get('MetaData_B', 'description'), 'Test metadata B')
        assert_array_equal(md.get('MetaData_C', 'value'), [0] * 15 + [1] * 3)
        assert_array_equal(md.get('MetaData_C', 'description'), 'Test metadata C')

    def test_set_get_overwrite(self):
        '''Test for MetaData.set() and MetaData.get(); overwriting values.'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1] * 10 + [0] * 5, 'Test metadata A')
        md.set('MetaData_B', [0] * 10 + [1] * 5, 'Test metadata B')

        md.set('MetaData_A', [10] * 10 + [0] * 5, 'Test metadata A')

        assert_array_equal(md.get('MetaData_A', 'value'), [10] * 10 + [0] * 5)
        assert_array_equal(md.get('MetaData_A', 'description'), 'Test metadata A')
        assert_array_equal(md.get('MetaData_B', 'value'), [0] * 10 + [1] * 5)
        assert_array_equal(md.get('MetaData_B', 'description'), 'Test metadata B')

    def test_set_get_overwrite_resize(self):
        '''Test for MetaData.set() and MetaData.get(); overwriting and resizing values.'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1, 1, 1, 0, 0], 'Test metadata A')
        md.set('MetaData_B', [0, 0, 0, 1, 1], 'Test metadata B')

        md.set('MetaData_A', [2, 2, 2, 0, 0, 1, 1], 'Test metadata A')

        assert_array_equal(md.get('MetaData_A', 'value'), [2, 2, 2, 0, 0, 1, 1])
        assert_array_equal(md.get('MetaData_A', 'description'), 'Test metadata A')
        assert_array_equal(md.get('MetaData_B', 'value'), [0, 0, 0, 1, 1, np.nan, np.nan])
        assert_array_equal(md.get('MetaData_B', 'description'), 'Test metadata B')

    def test_set_get_update(self):
        '''Test for MetaData.set() and MetaData.get(); updating values.'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1] * 10 + [0] * 5, 'Test metadata A')
        md.set('MetaData_B', [0] * 10 + [1] * 5, 'Test metadata B')

        md.set('MetaData_A', [10] * 10 + [0] * 5, 'Test metadata A', updater=lambda x, y: x + y)

        assert_array_equal(md.get('MetaData_A', 'value'), [11] * 10 + [0] * 5)
        assert_array_equal(md.get('MetaData_A', 'description'), 'Test metadata A')
        assert_array_equal(md.get('MetaData_B', 'value'), [0] * 10 + [1] * 5)
        assert_array_equal(md.get('MetaData_B', 'description'), 'Test metadata B')

    def test_get_notfound(self):
        '''Test for MetaData.get(); key not found case.'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1] * 10 + [0] * 5, 'Test metadata A')
        md.set('MetaData_B', [0] * 10 + [1] * 5, 'Test metadata B')

        assert_array_equal(md.get('MetaData_NotFound', 'value'), None)
        assert_array_equal(md.get('MetaData_NotFound', 'description'), None)

    def test_get_value_len(self):
        '''Test for get_value_len().'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1] * 10 + [0] * 5, 'Test metadata A')
        md.set('MetaData_B', [0] * 10 + [1] * 5, 'Test metadata B')

        assert_array_equal(md.get_value_len(), 15)

    def test_keylist(self):
        '''Test for keylist().'''
        md = metadata.MetaData()
        md.set('MetaData_A', [1] * 10 + [0] * 5, 'Test metadata A')
        md.set('MetaData_B', [0] * 10 + [1] * 5, 'Test metadata B')

        assert_array_equal(md.keylist(), ['MetaData_A', 'MetaData_B'])


if __name__ == '__main__':
    unittest.main()
