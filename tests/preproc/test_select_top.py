import unittest

import numpy as np

from bdpy.preproc.select_top import select_top


class TestPreprocessorSelectTop(unittest.TestCase):

    @classmethod
    def test_select_top_default(cls):
        '''Test for select_top (default, axis=0)'''

        test_data = np.array([[1,  2,  3,  4,  5],
                              [11, 12, 13, 14, 15],
                              [21, 22, 23, 24, 25],
                              [31, 32, 33, 34, 35],
                              [41, 42, 43, 44, 45]])
        test_value = np.array([15, 3, 6, 20, 0])
        test_num = 3

        exp_output_data = np.array([[1,  2,  3,  4,  5],
                                    [21, 22, 23, 24, 25],
                                    [31, 32, 33, 34, 35]])
        exp_output_index = np.array([0, 2, 3])

        test_output_data, test_output_index = select_top(
            test_data, test_value, test_num)

        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_index, exp_output_index)

    @classmethod
    def test_select_top_axisone(cls):
        '''Test for select_top (axis=1)'''

        test_data = np.array([[1,  2,  3,  4,  5],
                              [11, 12, 13, 14, 15],
                              [21, 22, 23, 24, 25],
                              [31, 32, 33, 34, 35],
                              [41, 42, 43, 44, 45]])
        test_value = np.array([15, 3, 6, 20, 0])
        test_num = 3

        exp_output_data = np.array([[1,  3,  4],
                                    [11, 13, 14],
                                    [21, 23, 24],
                                    [31, 33, 34],
                                    [41, 43, 44]])
        exp_output_index = np.array([0, 2, 3])

        test_output_data, test_output_index = select_top(
            test_data, test_value, test_num, axis=1)

        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_index, exp_output_index)