'''Tests for bdpy.ml'''


import unittest

import numpy as np

from bdpy.ml import regress


class TestRegress(unittest.TestCase):
    '''Tests for 'ml.regress' module'''

    def test_add_bias_default(self):
        '''Test of bdpy.ml.regress.add_bias (default: axis=0)'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120],
                          [200, 210, 220],
                          [1, 1, 1]])

        test_y = regress.add_bias(x)

        np.testing.assert_array_equal(test_y, exp_y)

    def test_add_bias_axisone(self):
        '''Test of bdpy.ml.regress.add_bias (axis=1)'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120, 1],
                          [200, 210, 220, 1]])

        test_y = regress.add_bias(x, axis=1)

        np.testing.assert_array_equal(test_y, exp_y)

    def test_add_bias_axiszero(self):
        '''Test of bdpy.ml.regress.add_bias (axis=0)'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120],
                          [200, 210, 220],
                          [1, 1, 1]])

        test_y = regress.add_bias(x, axis=0)

        np.testing.assert_array_equal(test_y, exp_y)

    def test_add_bias_invalidaxis(self):
        '''Exception test of bdpy.ml.regress.add_bias
           (invalid input in 'axis')'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        self.assertRaises(ValueError, lambda: regress.add_bias(x, axis=-1))


if __name__ == '__main__':
    unittest.main()