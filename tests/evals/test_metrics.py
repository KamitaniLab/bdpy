import unittest

import pickle
import warnings
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from bdpy.evals import metrics
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification


def identification_accuracy(pred, true, metric='correlation'):
    '''Pair-wise identification accuracy, computed straight from its definition.

    For each sample, count how many of the other candidates are closer to it
    than its own counterpart, using the same similarity `1 - distance` as
    `pairwise_identification`.
    '''
    d = 1 - cdist(
        pred.reshape(pred.shape[0], -1),
        true.reshape(true.shape[0], -1),
        metric=metric
    )
    n_sample = d.shape[0]
    return np.array([
        np.sum(np.delete(d[i], i) < d[i, i]) / (n_sample - 1)
        for i in range(n_sample)
    ])


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


class TestMetricsNanAndDegenerateInputs(unittest.TestCase):
    """Behavior around NaN and zero-variance inputs."""

    def test_profile_correlation_constant_column(self):
        # A column with zero variance has an undefined correlation.
        n_sample, n_feat = 10, 8
        rand = np.random.RandomState(0)
        x = rand.rand(n_sample, n_feat)
        y = rand.rand(n_sample, n_feat)
        x[:, 3] = 7.0

        r = profile_correlation(x, y).ravel()

        self.assertTrue(np.isnan(r[3]))
        np.testing.assert_allclose(
            np.delete(r, 3),
            [
                np.corrcoef(x[:, j], y[:, j])[0, 1]
                for j in range(n_feat) if j != 3
            ],
            rtol=1e-12, atol=1e-12
        )

    def test_pattern_correlation_constant_row(self):
        # A sample with zero variance has an undefined correlation.
        n_sample, n_feat = 6, 20
        rand = np.random.RandomState(1)
        x = rand.rand(n_sample, n_feat)
        y = rand.rand(n_sample, n_feat)
        x[2, :] = 5.0

        r = pattern_correlation(x, y)

        self.assertTrue(np.isnan(r[2]))
        np.testing.assert_allclose(
            np.delete(r, 2),
            [
                np.corrcoef(x[i, :], y[i, :])[0, 1]
                for i in range(n_sample) if i != 2
            ],
            rtol=1e-12, atol=1e-12
        )

    def test_pattern_correlation_removes_nan_columns(self):
        # A single NaN drops the whole column for every sample.
        n_sample, n_feat = 6, 20
        rand = np.random.RandomState(2)
        x = rand.rand(n_sample, n_feat)
        y = rand.rand(n_sample, n_feat)
        x[3, 7] = np.nan

        with self.assertWarns(UserWarning):
            r = pattern_correlation(x, y)

        xd = np.delete(x, 7, axis=1)
        yd = np.delete(y, 7, axis=1)
        np.testing.assert_allclose(
            r,
            [np.corrcoef(xd[i, :], yd[i, :])[0, 1] for i in range(n_sample)],
            rtol=1e-12, atol=1e-12
        )

    def test_pattern_correlation_keeps_nan_columns_when_disabled(self):
        n_sample, n_feat = 6, 20
        rand = np.random.RandomState(3)
        x = rand.rand(n_sample, n_feat)
        y = rand.rand(n_sample, n_feat)
        x[3, 7] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = pattern_correlation(x, y, remove_nan=False)

        self.assertFalse(any(
            'NaN column removed' in str(w.message) for w in caught
        ))
        self.assertTrue(np.isnan(r[3]))
        self.assertFalse(np.isnan(np.delete(r, 3)).any())

    def test_pattern_correlation_with_mean_and_std(self):
        n_sample, n_feat = 6, 20
        rand = np.random.RandomState(4)
        x = rand.rand(n_sample, n_feat)
        y = rand.rand(n_sample, n_feat)
        mean = rand.rand(n_feat)
        std = rand.rand(n_feat) + 0.5

        r = pattern_correlation(x, y, mean=mean, std=std)

        xs = (x - mean) / std
        ys = (y - mean) / std
        np.testing.assert_allclose(
            r,
            [np.corrcoef(xs[i, :], ys[i, :])[0, 1] for i in range(n_sample)],
            rtol=1e-12, atol=1e-12
        )

    def test_pairwise_identification_matches_definition(self):
        # Mix well-predicted and poorly-predicted samples so that the accuracy
        # is not degenerate, and include an exactly correlated and an exactly
        # anti-correlated sample.
        n_sample, n_feat = 12, 40
        rand = np.random.RandomState(5)
        true = rand.rand(n_sample, n_feat)
        pred = true + 5.0 * rand.rand(n_sample, n_feat)
        pred[0] = true[0]
        pred[1] = -true[1]

        cr = pairwise_identification(pred, true)

        self.assertEqual(cr.shape, (n_sample,))
        # Guard against a degenerate all-correct case.
        self.assertTrue(np.any((cr[2:] > 0) & (cr[2:] < 1)))
        np.testing.assert_allclose(
            cr, identification_accuracy(pred, true), rtol=1e-12, atol=1e-12
        )

    def test_pairwise_identification_constant_row(self):
        # A sample with zero variance gives an undefined correlation distance;
        # those comparisons are dropped instead of failing.
        n_sample, n_feat = 10, 30
        rand = np.random.RandomState(6)
        true = rand.rand(n_sample, n_feat)
        pred = true + 5.0 * rand.rand(n_sample, n_feat)
        pred[4, :] = 3.0

        with self.assertWarns(UserWarning):
            cr = pairwise_identification(pred, true)

        self.assertEqual(cr.shape, (n_sample,))
        np.testing.assert_allclose(
            np.delete(cr, 4),
            np.delete(identification_accuracy(pred, true), 4),
            rtol=1e-12, atol=1e-12
        )

    def test_pairwise_identification_non_correlation_metric(self):
        n_sample, n_feat = 10, 30
        rand = np.random.RandomState(7)
        true = rand.rand(n_sample, n_feat)
        pred = true + 5.0 * rand.rand(n_sample, n_feat)

        cr = pairwise_identification(pred, true, metric='euclidean')

        self.assertTrue(np.any((cr > 0) & (cr < 1)))
        np.testing.assert_allclose(
            cr, identification_accuracy(pred, true, metric='euclidean'),
            rtol=1e-12, atol=1e-12
        )

    def test_results_do_not_depend_on_the_block_size(self):
        # The metrics process the units in blocks to bound memory usage. Shrink
        # a block so that several of them are needed, with the NaN column
        # falling inside a block rather than on its boundary.
        n_sample, n_feat, nan_col = 8, 37, 17
        rand = np.random.RandomState(8)
        x = rand.rand(n_sample, n_feat)
        y = rand.rand(n_sample, n_feat)
        x[5, nan_col] = np.nan

        with patch.object(metrics, '_BLOCK_ELEMENTS', 3 * n_sample):
            r_prof = profile_correlation(x, y).ravel()
            with self.assertWarns(UserWarning):
                r_patt = pattern_correlation(x, y)
            with self.assertWarns(UserWarning):
                cr = pairwise_identification(x, y)

        self.assertTrue(np.isnan(r_prof[nan_col]))
        np.testing.assert_allclose(
            np.delete(r_prof, nan_col),
            [
                np.corrcoef(x[:, j], y[:, j])[0, 1]
                for j in range(n_feat) if j != nan_col
            ],
            rtol=1e-12, atol=1e-12
        )

        xd = np.delete(x, nan_col, axis=1)
        yd = np.delete(y, nan_col, axis=1)
        np.testing.assert_allclose(
            r_patt,
            [np.corrcoef(xd[i, :], yd[i, :])[0, 1] for i in range(n_sample)],
            rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            cr, identification_accuracy(xd, yd), rtol=1e-12, atol=1e-12
        )


if __name__ == '__main__':
    unittest.main()