'''bdpy.evals.metrics'''


import numpy as np
from scipy.spatial.distance import cdist


def profile_correlation(x, y):
    '''Profile correlation.'''

    sample_axis = 0

    orig_shape = x.shape
    n_sample = orig_shape[sample_axis]

    _x = x.reshape(n_sample, -1)
    _y = y.reshape(n_sample, -1)

    n_feat = _y.shape[1]

    r = np.array(
        [
            np.corrcoef(
                _x[:, j].flatten(),
                _y[:, j].flatten()
            )[0, 1]
            for j in range(n_feat)
        ]
    )

    r = r.reshape((1,) + orig_shape[1:])

    return r


def pattern_correlation(x, y, mean=None, std=None):
    '''Pattern correlation.'''

    sample_axis = 0

    orig_shape = x.shape
    n_sample = orig_shape[sample_axis]

    _x = x.reshape(n_sample, -1)
    _y = y.reshape(n_sample, -1)

    if mean is not None and std is not None:
        m = mean.reshape(-1)
        s = std.reshape(-1)

        _x = (_x - m) / s
        _y = (_y - m) / s
    
    r = np.array(
        [
            np.corrcoef(
                _x[i, :].flatten(),
                _y[i, :].flatten()
            )[0, 1]
            for i in range(n_sample)
        ]
    )

    return r


def pairwise_identification(pred, true, metric='correlation'):
    '''Pair-wise identification.'''

    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)

    d = 1 - cdist(p, t, metric=metric)

    cr = np.sum(d - np.diag(d)[:, np.newaxis] < 0, axis=1) / (d.shape[1] - 1)

    return cr
