# coding: utf-8
'''Tests for ml'''


import unittest

import numpy as np

from bdpy.ml.crossvalidation import cvindex_groupwise, make_cvindex, make_cvindex_generator


class TestCVIndexGroupwise(unittest.TestCase):

    def test_cvindex_groupwise(self):

        # Test data
        x = np.array([
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
            5, 5, 5,
            6, 6, 6
        ])

        # Expected output
        train_index = [
            np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        ]

        test_index = [
            np.array([ 0,  1,  2]),
            np.array([ 3,  4,  5]),
            np.array([ 6,  7,  8]),
            np.array([ 9, 10, 11]),
            np.array([12, 13, 14]),
            np.array([15, 16, 17])
        ]

        cvindex = cvindex_groupwise(x)

        for i, (tr, te) in enumerate(cvindex):
            self.assertTrue(np.array_equal(train_index[i], tr))
            self.assertTrue(np.array_equal(test_index[i], te))

    def test_cvindex_groupwise_exclusive(self):

        # Test data
        x = np.array([
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
            5, 5, 5,
            6, 6, 6
        ])

        # Exclusive labels
        a = np.array([
            1, 2, 3,
            4, 5, 6,
            1, 2, 3,
            4, 5, 6,
            1, 2, 3,
            4, 5, 6,
        ])

        # Expected output
        train_index = [
            np.array([3, 4, 5, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 6,  7,  8, 12, 13, 14]),
            np.array([3, 4, 5, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 6,  7,  8, 12, 13, 14]),
            np.array([3, 4, 5, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 6,  7,  8, 12, 13, 14])
        ]

        test_index = [
            np.array([ 0,  1,  2]),
            np.array([ 3,  4,  5]),
            np.array([ 6,  7,  8]),
            np.array([ 9, 10, 11]),
            np.array([12, 13, 14]),
            np.array([15, 16, 17])
        ]

        cvindex = cvindex_groupwise(x, exclusive=a)

        for i, (tr, te) in enumerate(cvindex):
            self.assertTrue(np.array_equal(train_index[i], tr))
            self.assertTrue(np.array_equal(test_index[i], te))


class TestMakeCVIndex(unittest.TestCase):
    def test_make_cvindex(self):
        # Test data
        x = np.array([1, 1, 2, 2, 3, 3])

        # Expected output
        train_index = np.array([
            [False,  True,  True],
            [False,  True,  True],
            [ True, False,  True],
            [ True, False,  True],
            [ True,  True, False],
            [ True,  True, False]])

        test_index = np.array([
            [ True, False, False],
            [ True, False, False],
            [False,  True, False],
            [False,  True, False],
            [False, False,  True],
            [False, False,  True]])

        result = make_cvindex(x)

        self.assertTrue(np.array_equal(train_index, result[0]))
        self.assertTrue(np.array_equal(test_index, result[1]))


class TestMakeCVIndexGenerator(unittest.TestCase):
    pass  # TODO: implement


if __name__ == '__main__':
    unittest.main()
