from bdpy.evals.metrics import profile_correlation, pattern_correlation


import unittest

import numpy as np


class TestEval(unittest.TestCase):
    def test_profile_correlation(self):
        # 2-d array
        n = 30
        x = np.random.rand(10, n)
        y = np.random.rand(10, n)
        r = np.array([[
            np.corrcoef(x[:, i], y[:, i])[0, 1]
            for i in range(n)
        ]])

        self.assertTrue(np.array_equal(
            profile_correlation(x, y), r
        ))
        self.assertEqual(profile_correlation(x, y).shape, (1, n))

        # Multi-d array
        x = np.random.rand(10, 4, 3, 2)
        y = np.random.rand(10, 4, 3, 2)
        xf = x.reshape(10, -1)
        yf = y.reshape(10, -1)
        r = np.array([[
            np.corrcoef(xf[:, i], yf[:, i])[0, 1]
            for i in range(4 * 3 * 2)
        ]])
        r = r.reshape(1, 4, 3, 2)

        self.assertTrue(np.array_equal(
            profile_correlation(x, y), r
        ))
        self.assertEqual(profile_correlation(x, y).shape, (1, 4, 3, 2))

    def test_pattern_correlation(self):
        # 2-d array
        x = np.random.rand(10, 30)
        y = np.random.rand(10, 30)
        r = np.array([
            np.corrcoef(x[i, :], y[i, :])[0, 1]
            for i in range(10)
        ])

        self.assertTrue(np.array_equal(
            pattern_correlation(x, y), r
        ))
        self.assertEqual(pattern_correlation(x, y).shape, (10,))

        # Multi-d array
        x = np.random.rand(10, 4, 3, 2)
        y = np.random.rand(10, 4, 3, 2)
        xf = x.reshape(10, -1)
        yf = y.reshape(10, -1)
        r = np.array([
            np.corrcoef(xf[i, :], yf[i, :])[0, 1]
            for i in range(10)
        ])

        self.assertTrue(np.array_equal(
            pattern_correlation(x, y), r
        ))
        self.assertEqual(pattern_correlation(x, y).shape, (10,))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEval)
    unittest.TextTestRunner(verbosity=2).run(suite)
