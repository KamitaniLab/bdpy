"""Tests for bdpy.preprocessor"""

import unittest
import numpy as np
from scipy.signal import detrend

from bdpy import preproc


class TestPreprocessor(unittest.TestCase):
    """Tests of 'preprocessor' module"""


    def test_average_sample_pass0001(self):
        """Test for average_sample (pass case 0001)"""

        x = np.random.rand(10, 100)
        g = np.array([ 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 ])
        
        exp_output_x = np.vstack(( np.average(x[0:5,:], axis = 0),
                                   np.average(x[5:10,:], axis = 0) ))
        exp_output_ind = np.array([ 0, 5 ])

        test_output_x, test_output_ind = preproc.average_sample(x, g, verbose = True)

        np.testing.assert_array_equal(test_output_x, exp_output_x)
        np.testing.assert_array_equal(test_output_ind, exp_output_ind)


    def test_detrend_sample_pass0001(self):
        """Test for detrend_sample (pass case 0001)"""

        x = np.random.rand(20, 10)
        g = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ])

        exp_output = np.vstack(( detrend(x[0:10, :], axis = 0, type = 'linear')
                                 + np.mean(x[0:10, :], axis = 0),
                                 detrend(x[10:20, :], axis = 0, type = 'linear')
                                 + np.mean(x[10:20, :], axis = 0) ))

        test_output = preproc.detrend_sample(x, g, verbose = True)
        
        np.testing.assert_array_equal(test_output, exp_output)


    def test_detrend_sample_pass0002(self):
        """Test for detrend_sample (pass case 0001)"""

        x = np.random.rand(20, 10)
        g = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ])

        exp_output = np.vstack((detrend(x[0:10, :], axis = 0, type = 'linear'),
                                detrend(x[10:20, :], axis = 0, type = 'linear')))

        test_output = preproc.detrend_sample(x, g, keep_mean=False, verbose=True)
        
        np.testing.assert_array_equal(test_output, exp_output)


    def test_normalize_sample_pass0001(self):
        """Test for normalize_sample (pass case 0001)"""

        x = np.random.rand(20, 10)
        g = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ])

        mA = np.mean(x[0:10, :], axis = 0)
        sA = np.std(x[0:10, :], axis = 0)
        mB = np.mean(x[10:20, :], axis = 0)
        sB = np.std(x[10:20, :], axis = 0)
        
        exp_output = np.vstack(( 100 * (x[0:10, :] - mA) / mA,
                                 100 * (x[10:20, :] - mB) / mB ))

        test_output = preproc.normalize_sample(x, g, verbose = True)
        
        np.testing.assert_array_equal(test_output, exp_output)


    def test_shift_sample_pass0001(self):
        """Test for shift_sample (pass case 0001)

        Single group, shift size = 1"""

        x = np.array([[  1,  2,  3 ],
                      [ 11, 12, 13 ],
                      [ 21, 22, 23 ],
                      [ 31, 32, 33 ],
                      [ 41, 42, 43 ]])
        grp = np.array([ 1, 1, 1, 1, 1 ])
        
        shift_size = 1

        # Default shift_size = 1

        exp_output_data = np.array([[ 11, 12, 13 ],
                                    [ 21, 22, 23 ],
                                    [ 31, 32, 33 ],
                                    [ 41, 42, 43 ]])
        exp_output_ind = [ 0, 1, 2, 3 ] 
        
        test_output_data, test_output_ind = preproc.shift_sample(x, grp, verbose = True)
        
        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_ind, exp_output_ind)


    def test_shift_sample_pass0002(self):
        """Test for shift_sample (pass case 0002)

        Two groups, shift size = 1"""

        x = np.array([[  1,  2,  3 ],
                      [ 11, 12, 13 ],
                      [ 21, 22, 23 ],
                      [ 31, 32, 33 ],
                      [ 41, 42, 43 ],
                      [ 51, 52, 53 ]])
        grp = np.array([ 1, 1, 1, 2, 2, 2 ])
        
        shift_size = 1

        # Default shift_size = 1

        exp_output_data = np.array([[ 11, 12, 13 ],
                                    [ 21, 22, 23 ],
                                    [ 41, 42, 43 ],
                                    [ 51, 52, 53 ]])
        exp_output_ind = [ 0, 1, 3, 4 ] 
        
        test_output_data, test_output_ind = preproc.shift_sample(x, grp, verbose = True)
        
        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_ind, exp_output_ind)


    def test_select_top_pass0001(self):
        """Test for select_top (default, axis=0)"""

        test_data = np.array([[  1,  2,  3,  4,  5 ],
                              [ 11, 12, 13, 14, 15 ],
                              [ 21, 22, 23, 24, 25 ],
                              [ 31, 32, 33, 34, 35 ],
                              [ 41, 42, 43, 44, 45 ]])
        test_value = np.array([ 15,  3,  6, 20,  0])
        test_num = 3;

        exp_output_data = np.array([[  1,  2,  3,  4,  5 ],
                                    [ 21, 22, 23, 24, 25 ],
                                    [ 31, 32, 33, 34, 35 ]])
        exp_output_index = np.array([0, 2, 3])

        test_output_data, test_output_index = preproc.select_top(test_data, test_value, test_num);

        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_index, exp_output_index)


    def test_select_top_pass0002(self):
        """Test for select_top (axis=1)"""

        test_data = np.array([[  1,  2,  3,  4,  5 ],
                              [ 11, 12, 13, 14, 15 ],
                              [ 21, 22, 23, 24, 25 ],
                              [ 31, 32, 33, 34, 35 ],
                              [ 41, 42, 43, 44, 45 ]])
        test_value = np.array([ 15,  3,  6, 20,  0])
        test_num = 3;

        exp_output_data = np.array([[  1,  3,  4 ],
                                    [ 11, 13, 14 ],
                                    [ 21, 23, 24 ],
                                    [ 31, 33, 34 ],
                                    [ 41, 43, 44 ]])
        exp_output_index = np.array([0, 2, 3])

        test_output_data, test_output_index = preproc.select_top(test_data, test_value, test_num, axis=1);

        np.testing.assert_array_equal(test_output_data, exp_output_data)
        np.testing.assert_array_equal(test_output_index, exp_output_index)


if __name__ == "__main__":
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessor)
    unittest.TextTestRunner(verbosity = 2).run(test_suite)
