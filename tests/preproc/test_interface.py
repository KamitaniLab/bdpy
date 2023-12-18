'''Tests for bdpy.preprocessor'''

import unittest

import numpy as np
from scipy.signal import detrend

from bdpy.preproc import interface


class TestPreprocessorInterface(unittest.TestCase):
    '''Tests of 'preprocessor' module'''

    @classmethod
    def test_average_sample(cls):
        '''Test for average_sample'''

        x = np.random.rand(10, 100)
        group = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        exp_output_x = np.vstack((np.average(x[0:5, :], axis=0),
                                  np.average(x[5:10, :], axis=0)))
        exp_output_ind = np.array([0, 5])

        test_output_x, test_output_ind = interface.average_sample(
            x, group, verbose=True)

        np.testing.assert_array_equal(test_output_x, exp_output_x)
        np.testing.assert_array_equal(test_output_ind, exp_output_ind)

    @classmethod
    def test_detrend_sample_default(cls):
        '''Test for detrend_sample (default)'''

        x = np.random.rand(20, 10)
        group = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        exp_output = np.vstack((detrend(x[0:10, :], axis=0, type='linear')
                                + np.mean(x[0:10, :], axis=0),
                                detrend(x[10:20, :], axis=0, type='linear')
                                + np.mean(x[10:20, :], axis=0)))

        test_output = interface.detrend_sample(x, group, verbose=True)

        np.testing.assert_array_equal(test_output, exp_output)

    @classmethod
    def test_detrend_sample_nokeepmean(cls):
        '''Test for detrend_sample (keep_mean=False)'''

        x = np.random.rand(20, 10)
        group = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        exp_output = np.vstack((detrend(x[0:10, :], axis=0, type='linear'),
                                detrend(x[10:20, :], axis=0, type='linear')))

        test_output = interface.detrend_sample(
            x, group, keep_mean=False, verbose=True)

        np.testing.assert_array_equal(test_output, exp_output)

    @classmethod
    def test_normalize_sample(cls):
        '''Test for normalize_sample (default)'''

        x = np.random.rand(20, 10)
        group = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        mean_a = np.mean(x[0:10, :], axis=0)
        mean_b = np.mean(x[10:20, :], axis=0)

        exp_output = np.vstack((100 * (x[0:10, :] - mean_a) / mean_a,
                                100 * (x[10:20, :] - mean_b) / mean_b))

        test_output = interface.normalize_sample(x, group, verbose=True)

        np.testing.assert_array_equal(test_output, exp_output)

    @classmethod
    def test_shift_sample_singlegroup(cls):
        '''Test for shift_sample (single group, shift_size=1)'''

        x = np.array([[1,  2,  3],
                      [11, 12, 13],
                      [21, 22, 23],
                      [31, 32, 33],
                      [41, 42, 43]])
        grp = np.array([1, 1, 1, 1, 1])

        exp_output_data = np.array([[11, 12, 13],
                                    [21, 22, 23],
                                    [31, 32, 33],
                                    [41, 42, 43]])
        exp_output_ind = [0, 1, 2, 3]

        # Default shift_size = 1
        test_output_data, test_output_ind = interface.shift_sample(
            x, grp, verbose=True)

        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_ind, exp_output_ind)

    @classmethod
    def test_shift_sample_twogroup(cls):
        '''Test for shift_sample (two groups, shift_size=1)'''

        x = np.array([[1,  2,  3],
                      [11, 12, 13],
                      [21, 22, 23],
                      [31, 32, 33],
                      [41, 42, 43],
                      [51, 52, 53]])
        grp = np.array([1, 1, 1, 2, 2, 2])

        exp_output_data = np.array([[11, 12, 13],
                                    [21, 22, 23],
                                    [41, 42, 43],
                                    [51, 52, 53]])
        exp_output_ind = [0, 1, 3, 4]

        # Default shift_size=1
        test_output_data, test_output_ind = interface.shift_sample(
            x, grp, verbose=True)

        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_ind, exp_output_ind)


if __name__ == '__main__':
    unittest.main()