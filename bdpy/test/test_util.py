"""Tests for bdpy.util"""

import unittest
import numpy as np

import bdpy


class TestUtil(unittest.TestCase):
    """Tests for 'util' module"""

    
    def test_create_groupvector_pass0001(self):
        """Test for create_groupvector (pass case 0001: list and scalar input)"""

        x = [ 1, 2, 3 ]
        y = 2

        exp_output = [ 1, 1, 2, 2, 3, 3 ]

        test_output = bdpy.util.create_groupvector(x, y)

        self.assertTrue((test_output == exp_output).all())


    def test_create_groupvector_pass0002(self):
        """Test for create_groupvector (pass case 0002: list and list input)"""

        x = [ 1, 2, 3 ]
        y = [ 2, 4, 2 ]

        exp_output = [ 1, 1, 2, 2, 2, 2, 3, 3 ]

        test_output = bdpy.util.create_groupvector(x, y)

        self.assertTrue((test_output == exp_output).all())


    def test_create_groupvector_pass0003(self):
        """Test for create_groupvector (pass case 0003: np.ndarray and scalar input)"""

        x = np.array([ 1, 2, 3 ])
        y = 2

        exp_output = np.array([ 1, 1, 2, 2, 3, 3 ])

        test_output = bdpy.util.create_groupvector(x, y)

        np.testing.assert_array_equal(test_output, exp_output)


    def test_create_groupvector_pass0005(self):
        """Test for create_groupvector (pass case 0004: np.ndarray and np.ndarray input)"""

        x = np.array([ 1, 2, 3 ])
        y = np.array([ 2, 4, 2 ])

        exp_output = np.array([ 1, 1, 2, 2, 2, 2, 3, 3 ])

        test_output = bdpy.util.create_groupvector(x, y)

        np.testing.assert_array_equal(test_output, exp_output)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtil)
    unittest.TextTestRunner(verbosity = 2).run(suite)
