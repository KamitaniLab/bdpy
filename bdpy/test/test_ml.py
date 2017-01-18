# coding: utf-8
"""Tests for bdpy.util"""

import sys
import os
import unittest

import numpy as np

from bdpy import ml


class TestMl(unittest.TestCase):
    """Tests for 'util' module"""

    
    def test_make_crossvalidationindex_pass0001(self):
        """Test for make_crossvalidationindex (pass case 0001)"""

        test_input = np.array([ 1, 1, 2, 2, 3, 3 ])

        exp_output_a = np.array([[ False, True,  True  ],
                                 [ False, True,  True  ],
                                 [ True,  False, True  ],
                                 [ True,  False, True  ],
                                 [ True,  True,  False ],
                                 [ True,  True,  False ]])
        exp_output_b = np.array([[ True,  False, False ],
                                 [ True,  False, False ],
                                 [ False, True,  False ],
                                 [ False, True,  False ],
                                 [ False, False, True  ],
                                 [ False, False, True  ]])

        test_output_a, test_output_b = ml.make_crossvalidationindex(test_input)

        self.assertTrue((test_output_a == exp_output_a).all())
        self.assertTrue((test_output_b == exp_output_b).all())


    def test_add_bias_pass0001(self):
        """Test of bdpy.ml.regress.add_bias (default: axis=0)"""
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120],
                          [200, 210, 220],
                          [1, 1, 1]])

        test_y = ml.add_bias(x)

        np.testing.assert_array_equal(test_y, exp_y)


    def test_add_bias_pass0002(self):
        """Test of bdpy.ml.regress.add_bias (axis=1)"""
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120, 1],
                          [200, 210, 220, 1]])

        test_y = ml.add_bias(x, axis=1)

        np.testing.assert_array_equal(test_y, exp_y)


    def test_ensemble_get_majority(self):
        """
        Tests of 'classifier.emsenble.get_majority' module
        """
        data = np.array([[1, 3, 2, 1, 2],
                         [2, 1, 0, 0, 2],
                         [2, 1, 1, 0, 2],
                         [1, 3, 3, 1, 1],
                         [0, 2, 3, 3, 0],
                         [3, 2, 2, 2, 1],
                         [3, 1, 3, 2, 0],
                         [3, 2, 0, 3, 1]])
        # 列ごと，行ごとに多数決した場合の解
        # 同数の場合は，小さい方の数値が採用される
        ans_by_column = np.array([3, 1, 3, 0, 1])
        ans_by_row = np.array([1, 0, 1, 1, 0, 2, 3, 3])
        np.testing.assert_array_almost_equal(ml.get_majority(data, axis = 0), ans_by_column)
        np.testing.assert_array_almost_equal(ml.get_majority(data, axis = 1), ans_by_row)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMl)
    unittest.TextTestRunner(verbosity = 2).run(suite)
