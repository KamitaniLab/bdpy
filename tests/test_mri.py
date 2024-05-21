'''Tests for bdpy.util'''

import os
import unittest

import numpy as np
import scipy.io as sio

import bdpy.mri as bmr


class TestMri(unittest.TestCase):
    '''Tests for 'mri' module'''
    def setUp(self) -> None:
        pass

    def test_get_roiflag_pass0001(self):
        '''Test for get_roiflag (pass case 0001)'''

        roi_xyz = [np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]])]
        epi_xyz = np.array([[1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6]])

        exp_output = np.array([1, 1, 1, 0, 0, 0])

        test_output = bmr.get_roiflag(roi_xyz, epi_xyz)

        self.assertTrue((test_output == exp_output).all())

    def test_get_roiflag_pass0002(self):
        '''Test for get_roiflag (pass case 0002)'''

        roi_xyz = [np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]]),
                   np.array([[5, 6],
                             [5, 6],
                             [5, 6]])]
        epi_xyz = np.array([[1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6]])

        exp_output = np.array([[1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1]])

        test_output = bmr.get_roiflag(roi_xyz, epi_xyz)

        self.assertTrue((test_output == exp_output).all())


if __name__ == '__main__':
    unittest.main()
