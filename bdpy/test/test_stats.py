"""
Tests for bdpy.stats

Tested functions:

- corrcoef
- corrmat
"""


import unittest
import numpy as np

import bdpy.stats as bdst


class TestStats(unittest.TestCase):
    """Tests for bdpy.stats"""

    
    def test_corrcoef_0000(self):
        """Test for corrcoef (matrix and matrix, default, var=row)"""

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.diag(np.corrcoef(x, y)[:x.shape[0], x.shape[0]:])

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)


    def test_corrcoef_0001(self):
        """Test for corrcoef (matrix and matrix, var=col)"""

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.diag(np.corrcoef(x, y, rowvar=0)[:x.shape[1], x.shape[1]:])

        test_output = bdst.corrcoef(x, y, var='col')

        np.testing.assert_array_equal(test_output, exp_output)


    def test_corrcoef_0050(self):
        """Test for corrcoef (vector and vector)"""

        x = np.random.rand(100)
        y = np.random.rand(100)

        exp_output = np.corrcoef(x, y)[0, 1]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)


    def test_corrcoef_0051(self):
        """Test for corrcoef (horizontal vector and horizontal vector)"""

        x = np.random.rand(1, 100)
        y = np.random.rand(1, 100)

        exp_output = np.corrcoef(x, y)[0, 1]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)


    def test_corrcoef_0052(self):
        """Test for corrcoef (vertical vector and vertical vector)"""

        x = np.random.rand(100, 1)
        y = np.random.rand(100, 1)

        exp_output = np.corrcoef(x.T, y.T)[0, 1]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)


    def test_corrcoef_0100(self):
        """Test for corrcoef (matrix and vector, var=row)"""

        x = np.random.rand(100, 10)
        y = np.random.rand(10)

        exp_output = np.corrcoef(y, x)[0, 1:]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_almost_equal(test_output, exp_output)


    def test_corrcoef_0101(self):
        """Test for corrcoef (matrix and vector, var=col)"""

        x = np.random.rand(100, 10)
        y = np.random.rand(100)

        exp_output = np.corrcoef(y, x, rowvar=0)[0, 1:]

        test_output = bdst.corrcoef(x, y, var='col')

        np.testing.assert_array_almost_equal(test_output, exp_output)


    def test_corrcoef_0110(self):
        """Test for corrcoef (vector and matrix, var=row)"""

        x = np.random.rand(10)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y)[0, 1:]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_almost_equal(test_output, exp_output)


    def test_corrcoef_0111(self):
        """Test for corrcoef (vector and matrix, var=col)"""

        x = np.random.rand(100)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y, rowvar=0)[0, 1:]

        test_output = bdst.corrcoef(x, y, var='col')

        np.testing.assert_array_almost_equal(test_output, exp_output)


    def test_corrmat_0000(self):
        """Test for corrmat (default, var=row)"""

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y)[:x.shape[0], x.shape[0]:]

        test_output = bdst.corrmat(x, y)

        np.testing.assert_array_almost_equal(test_output, exp_output)


    def test_corrmat_0001(self):
        """Test for corrmat (var=col)"""

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y, rowvar=0)[:x.shape[1], x.shape[1]:]

        test_output = bdst.corrmat(x, y, var='col')

        np.testing.assert_array_almost_equal(test_output, exp_output)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStats)
    unittest.TextTestRunner(verbosity = 2).run(suite)
