from unittest import TestCase, TestLoader, TextTestRunner

import bdpy
import numpy as np


class TestVmap(TestCase):

    #     # Add value-label map
    #     label_map_add = {(i + 100): 'label-%04d-add' % i for i in range(n_sample)}
    #     label_map_new = label_map.update(label_map_add)

    #     # bdata.add_vmap('Label', label_map_add)
    #     # print(bdata.get_vmap('Label'))
    #     # print(label_map)
    #     # assert bdata.get_vmap('Label') == label_map_new

    def test_vmap_add_get(self):
        bdata = bdpy.BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4'}
        label = ['label-1', 'label-2', 'label-3', 'label-4']
        
        bdata.add_vmap('Label', label_map)
        assert bdata.get_vmap('Label') == label_map

        # Get labels
        np.testing.assert_array_equal(bdata.get_label('Label'), label)

    def test_vmap_add_same_map(self):
        bdata = bdpy.BData()
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
        
        bdata = bdpy.BData()
        bdata.add(np.random.rand(n_sample, 3), 'MainData')
        bdata.add(np.arange(n_sample) + 1, 'Label')

        label_map = {(i + 1): 'label-%04d' % (i + 1) for i in range(n_sample)}
        label = ['label-%04d' % (i + 1) for i in range(n_sample)]
        
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
        label_map_inconsist = {i: 'label-%04d-inconsist' % i for i in range(n_sample)}
        with self.assertRaises(ValueError):
            bdata.add_vmap('Label', label_map_inconsist)

    def test_vmap_add_unnecessary_vmap(self):
        bdata = bdpy.BData()
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
        bdata = bdpy.BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3'}

        with self.assertRaises(ValueError):
            bdata.add_vmap('Label', label_map)


if __name__ == "__main__":
    test_suite = TestLoader().loadTestsFromTestCase(TestVmap)
    TextTestRunner(verbosity=2).run(test_suite)
