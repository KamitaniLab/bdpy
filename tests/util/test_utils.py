'''Tests for bdpy.util'''


import unittest

import numpy as np

from bdpy.util import utils


class TestCreateGroupVector(unittest.TestCase):
    '''Tests for 'utils.create_groupvector'''

    def test_create_groupvector_pass0001(self):
        '''Test for create_groupvector (list and scalar inputs).'''

        x = [1, 2, 3]
        y = 2

        exp_output = [1, 1, 2, 2, 3, 3]

        test_output = utils.create_groupvector(x, y)

        self.assertTrue((test_output == exp_output).all())

    def test_create_groupvector_pass0002(self):
        '''Test for create_groupvector (list and list inputs).'''

        x = [1, 2, 3]
        y = [2, 4, 2]

        exp_output = [1, 1, 2, 2, 2, 2, 3, 3]

        test_output = utils.create_groupvector(x, y)

        self.assertTrue((test_output == exp_output).all())

    def test_create_groupvector_pass0003(self):
        '''Test for create_groupvector (Numpy array and scalar inputs).'''

        x = np.array([1, 2, 3])
        y = 2

        exp_output = np.array([1, 1, 2, 2, 3, 3])

        test_output = utils.create_groupvector(x, y)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_create_groupvector_pass0005(self):
        '''Test for create_groupvector (Numpy arrays inputs).'''

        x = np.array([1, 2, 3])
        y = np.array([2, 4, 2])

        exp_output = np.array([1, 1, 2, 2, 2, 2, 3, 3])

        test_output = utils.create_groupvector(x, y)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_create_groupvector_error(self):
        '''Test for create_groupvector (ValueError).'''

        x = [1, 2, 3]
        y = [0]

        self.assertRaises(ValueError, utils.create_groupvector, x, y)


class TestDivideChunks(unittest.TestCase):

    def test_divide_chunks(self):
        '''Test for divide_chunks.'''

        a = [1, 2, 3, 4, 5, 6, 7]

        # Test 1
        expected = [[1, 2, 3, 4],
                    [5, 6, 7]]
        actual = utils.divide_chunks(a, chunk_size=4)
        self.assertEqual(actual, expected)

        # Test 2
        expected = [[1, 2, 3],
                    [4, 5, 6],
                    [7]]
        actual = utils.divide_chunks(a, chunk_size=3)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
