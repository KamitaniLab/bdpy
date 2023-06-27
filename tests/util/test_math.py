import unittest

import numpy as np

from bdpy.util.math import average_elemwise


class TestMath(unittest.TestCase):

    def test_average_elemwise(self):
        a = np.array([1, 2, 3])
        b = np.array([9, 8, 7])
        ans_valid = np.array([5, 5, 5])
        ans_test = average_elemwise([a, b])
        np.testing.assert_array_equal(ans_test, ans_valid)

        a = np.array([[1, 2, 3]])
        b = np.array([[9, 8, 7]])
        ans_valid = np.array([5, 5, 5])
        ans_test = average_elemwise([a, b])
        np.testing.assert_array_equal(ans_test, ans_valid)

        a = np.array([[1, 2, 3]])
        b = np.array([9, 8, 7])
        ans_valid = np.array([5, 5, 5])
        ans_test = average_elemwise([a, b])
        np.testing.assert_array_equal(ans_test, ans_valid)

        a = np.array([1, 2, 3])
        b = np.array([[9, 8, 7]])
        ans_valid = np.array([5, 5, 5])
        ans_test = average_elemwise([a, b])
        np.testing.assert_array_equal(ans_test, ans_valid)

    def test_average_elemwise_keepdims(self):
        a = np.array([1, 2, 3])
        b = np.array([9, 8, 7])
        ans_valid = np.array([5, 5, 5])
        ans_test = average_elemwise([a, b], keepdims=True)
        np.testing.assert_array_equal(ans_test, ans_valid)

        a = np.array([[1, 2, 3]])
        b = np.array([[9, 8, 7]])
        ans_valid = np.array([[5, 5, 5]])
        ans_test = average_elemwise([a, b], keepdims=True)
        np.testing.assert_array_equal(ans_test, ans_valid)

        a = np.array([[1, 2, 3]])
        b = np.array([9, 8, 7])
        ans_valid = np.array([[5, 5, 5]])
        ans_test = average_elemwise([a, b], keepdims=True)
        np.testing.assert_array_equal(ans_test, ans_valid)

        a = np.array([1, 2, 3])
        b = np.array([[9, 8, 7]])
        ans_valid = np.array([[5, 5, 5]])
        ans_test = average_elemwise([a, b], keepdims=True)
        np.testing.assert_array_equal(ans_test, ans_valid)


if __name__ == '__main__':
    unittest.main()