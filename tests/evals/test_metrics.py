import unittest

import pickle

import numpy as np

from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification


class TestMetrics(unittest.TestCase):
    def test_profile_correlation(self):
        # 2-d array
        n = 30
        x = np.random.rand(10, n)
        y = np.random.rand(10, n)
        r = np.array([[
            np.corrcoef(x[:, i], y[:, i])[0, 1]
            for i in range(n)
        ]])

        np.testing.assert_allclose(
            profile_correlation(x, y), r, rtol=1e-12, atol=1e-12
        )
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

        np.testing.assert_allclose(
            profile_correlation(x, y), r, rtol=1e-12, atol=1e-12
        )
        self.assertEqual(profile_correlation(x, y).shape, (1, 4, 3, 2))

    def test_pattern_correlation(self):
        # 2-d array
        x = np.random.rand(10, 30)
        y = np.random.rand(10, 30)
        r = np.array([
            np.corrcoef(x[i, :], y[i, :])[0, 1]
            for i in range(10)
        ])

        np.testing.assert_allclose(
            pattern_correlation(x, y), r, rtol=1e-12, atol=1e-12
        )
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

        np.testing.assert_allclose(
            pattern_correlation(x, y), r, rtol=1e-12, atol=1e-12
        )
        self.assertEqual(pattern_correlation(x, y).shape, (10,))

    def test_2d(self):
        with open('tests/data/testdata-2d.pkl.gz', 'rb') as f:
            d = pickle.load(f)
        np.testing.assert_allclose(
            profile_correlation(d['x'], d['y']), d['r_prof'], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            pattern_correlation(d['x'], d['y']), d['r_patt'], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            pairwise_identification(d['x'], d['y']), d['ident_acc'], rtol=1e-12, atol=1e-12
        )

    def test_2d_nan(self):
        with open('tests/data/testdata-2d-nan.pkl.gz', 'rb') as f:
            d = pickle.load(f)
        # self.assertTrue(np.array_equal(
        #     profile_correlation(d['x'], d['y']),
        #     d['r_prof']
        # ))
        np.testing.assert_allclose(
            pattern_correlation(d['x'], d['y'], remove_nan=True),
            d['r_patt'], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            pairwise_identification(d['x'], d['y'], remove_nan=True),
            d['ident_acc'], rtol=1e-12, atol=1e-12
        )

if __name__ == '__main__':
    unittest.main()