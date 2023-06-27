import unittest

import numpy as np

from bdpy.ml import ensemble


class TestEnsemble(unittest.TestCase):
    def test_ensemble_get_majority(self):
        '''Tests of bdpy.ml.emsenble.get_majority'''
        data = np.array([[1, 3, 2, 1, 2],
                         [2, 1, 0, 0, 2],
                         [2, 1, 1, 0, 2],
                         [1, 3, 3, 1, 1],
                         [0, 2, 3, 3, 0],
                         [3, 2, 2, 2, 1],
                         [3, 1, 3, 2, 0],
                         [3, 2, 0, 3, 1]])
        # Get the major elements in each colum (axis=0) or row (axis=1).
        # The element with the smallest value will be returned when several
        # elements were the majority.
        ans_by_column = np.array([3, 1, 3, 0, 1])
        ans_by_row = np.array([1, 0, 1, 1, 0, 2, 3, 3])
        np.testing.assert_array_almost_equal(
            ensemble.get_majority(data, axis=0), ans_by_column)
        np.testing.assert_array_almost_equal(
            ensemble.get_majority(data, axis=1), ans_by_row)


if __name__ == '__main__':
    unittest.main()
