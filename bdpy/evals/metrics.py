"""bdpy.evals.metrics"""

import warnings
from typing import Iterator, Optional

import numpy as np
from scipy.spatial.distance import cdist

# Upper bound on the number of array elements held by a single working block
# (about 32 MiB in float64). The correlation metrics below work on blocks of
# whichever axis they do not reduce over, so that peak memory stays bounded
# even when the feature dimension reaches several hundred thousand.
_BLOCK_ELEMENTS = 2 ** 22


def _index_blocks(length: int, other_length: int) -> Iterator[slice]:
    """Split an axis into slices small enough to be processed at once.

    Parameters
    ----------
    length : int
        Number of indices to split.
    other_length : int
        Length of the other axis of the arrays to be processed.

    Yields
    ------
    slice
        Index range holding at most ``_BLOCK_ELEMENTS`` elements.
    """
    step = max(1, _BLOCK_ELEMENTS // max(other_length, 1))
    for start in range(0, length, step):
        yield slice(start, min(start + step, length))


def _nan_column_mask(
    x: np.ndarray,
    y: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> np.ndarray:
    """Flag the columns holding a NaN in either array.

    Parameters
    ----------
    x, y : numpy.ndarray
        Arrays of shape (n_samples, n_units).
    mean, std : numpy.ndarray or None
        Flattened standardization parameters, applied before the check when
        both are given.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape (n_units,).
    """
    n_sample, n_feat = x.shape
    nan_cols = np.empty(n_feat, dtype=bool)

    for cols in _index_blocks(n_feat, n_sample):
        xb = x[:, cols]
        yb = y[:, cols]
        if mean is not None and std is not None:
            xb = (xb - mean[cols]) / std[cols]
            yb = (yb - mean[cols]) / std[cols]
        nan_cols[cols] = np.isnan(xb).any(axis=0) | np.isnan(yb).any(axis=0)

    return nan_cols


def _column_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlate `x` and `y` column by column, over whole columns at once.

    Parameters
    ----------
    x, y : numpy.ndarray
        Arrays of shape (n_samples, n_units).

    Returns
    -------
    numpy.ndarray
        Correlation of each column, of shape (n_units,). A column with zero
        variance gives NaN, as ``numpy.corrcoef`` does.
    """
    n_sample, n_feat = x.shape
    r = np.empty(n_feat, dtype=np.float64)

    for cols in _index_blocks(n_feat, n_sample):
        xb = np.array(x[:, cols], dtype=np.float64)
        yb = np.array(y[:, cols], dtype=np.float64)
        xb -= xb.mean(axis=0)
        yb -= yb.mean(axis=0)

        with np.errstate(invalid='ignore', divide='ignore'):
            rb = np.einsum('ij,ij->j', xb, yb) / np.sqrt(np.einsum('ij,ij->j', xb, xb))
            rb /= np.sqrt(np.einsum('ij,ij->j', yb, yb))

        r[cols] = rb

    # numpy.corrcoef clips its result, so that a rounding error cannot push a
    # perfect correlation past 1.
    return np.clip(r, -1.0, 1.0)


def _row_correlation(
    x: np.ndarray,
    y: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    keep_cols: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Correlate `x` and `y` row by row, over whole rows at once.

    Parameters
    ----------
    x, y : numpy.ndarray
        Arrays of shape (n_samples, n_units).
    mean, std : numpy.ndarray or None
        Flattened standardization parameters, applied when both are given.
    keep_cols : numpy.ndarray or None
        Boolean mask of the columns to keep, applied after standardization.

    Returns
    -------
    numpy.ndarray
        Correlation of each row, of shape (n_samples,). A row with zero
        variance gives NaN, as ``numpy.corrcoef`` does.
    """
    n_sample, n_feat = x.shape
    r = np.empty(n_sample, dtype=np.float64)

    for rows in _index_blocks(n_sample, n_feat):
        xb = np.array(x[rows], dtype=np.float64)
        yb = np.array(y[rows], dtype=np.float64)

        if mean is not None and std is not None:
            xb -= mean
            xb /= std
            yb -= mean
            yb /= std

        if keep_cols is not None:
            xb = xb[:, keep_cols]
            yb = yb[:, keep_cols]

        xb -= xb.mean(axis=1, keepdims=True)
        yb -= yb.mean(axis=1, keepdims=True)

        with np.errstate(invalid='ignore', divide='ignore'):
            rb = np.einsum('ij,ij->i', xb, yb) / np.sqrt(np.einsum('ij,ij->i', xb, xb))
            rb /= np.sqrt(np.einsum('ij,ij->i', yb, yb))

        r[rows] = rb

    # numpy.corrcoef clips its result, so that a rounding error cannot push a
    # perfect correlation past 1.
    return np.clip(r, -1.0, 1.0)


def _correlation_similarity(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Correlate every row of `p` with every row of `t`.

    Equivalent to ``1 - cdist(p, t, metric='correlation')``, but computed as
    matrix products instead of one pass per pair.

    Parameters
    ----------
    p, t : numpy.ndarray
        Arrays of shape (n_samples, n_units).

    Returns
    -------
    numpy.ndarray
        Correlation matrix of shape (n_samples of `p`, n_samples of `t`). A row
        with zero variance gives NaN, as the correlation distance does.
    """
    n_p, n_feat = p.shape
    n_t = t.shape[0]
    n_row = max(n_p, n_t)

    sum_p = np.zeros(n_p)
    sum_t = np.zeros(n_t)
    for cols in _index_blocks(n_feat, n_row):
        sum_p += p[:, cols].sum(axis=1, dtype=np.float64)
        sum_t += t[:, cols].sum(axis=1, dtype=np.float64)
    mean_p = sum_p / n_feat
    mean_t = sum_t / n_feat

    d = np.zeros((n_p, n_t))
    sq_p = np.zeros(n_p)
    sq_t = np.zeros(n_t)
    for cols in _index_blocks(n_feat, n_row):
        pb = np.array(p[:, cols], dtype=np.float64) - mean_p[:, np.newaxis]
        tb = np.array(t[:, cols], dtype=np.float64) - mean_t[:, np.newaxis]
        d += pb @ tb.T
        sq_p += np.einsum('ij,ij->i', pb, pb)
        sq_t += np.einsum('ij,ij->i', tb, tb)

    with np.errstate(invalid='ignore', divide='ignore'):
        d /= np.sqrt(sq_p)[:, np.newaxis] * np.sqrt(sq_t)[np.newaxis, :]

    # scipy clips the cosine it derives the correlation distance from, so that
    # a rounding error cannot push a perfect correlation past 1.
    return np.clip(d, -1.0, 1.0)


def profile_correlation(x, y):
    """Profile correlation."""
    sample_axis = 0

    orig_shape = x.shape
    n_sample = orig_shape[sample_axis]

    _x = x.reshape(n_sample, -1)
    _y = y.reshape(n_sample, -1)

    r = _column_correlation(_x, _y)

    r = r.reshape((1,) + orig_shape[1:])

    return r


def pattern_correlation(x, y, mean=None, std=None, remove_nan=True):
    """Pattern correlation."""
    sample_axis = 0

    orig_shape = x.shape
    n_sample = orig_shape[sample_axis]

    _x = x.reshape(n_sample, -1)
    _y = y.reshape(n_sample, -1)

    m = None
    s = None
    if mean is not None and std is not None:
        m = mean.reshape(-1)
        s = std.reshape(-1)

    keep_cols = None
    if remove_nan:
        # Remove nan columns based on the decoded features
        nan_cols = _nan_column_mask(_x, _y, m, s)
        if nan_cols.any():
            warnings.warn('NaN column removed ({})'.format(np.sum(nan_cols)))
            keep_cols = ~nan_cols

    r = _row_correlation(_x, _y, m, s, keep_cols)

    return r


def pattern_cross_correlation(x, y, mean=None, std=None, remove_nan=True):
    """Pattern correlation.
    Output: cross correlation of size (n_sample, n_sample).
    The (i,j) element of r corresponds to the correlation between i-th row of x and j-th row of y.
    """
    sample_axis = 0

    orig_shape = x.shape
    n_sample = orig_shape[sample_axis]

    _x = x.reshape(n_sample, -1)
    _y = y.reshape(n_sample, -1)

    if mean is not None and std is not None:
        if mean.shape[sample_axis] == n_sample:
            # if mean and std are different across samples
            m = mean.reshape(n_sample, -1)
            s = std.reshape(n_sample, -1)
        else:
            m = mean.reshape(-1)
            s = std.reshape(-1)
        
        _x = (_x - m) / s
        _y = (_y - m) / s

    if remove_nan:
        # Remove nan columns based on the decoded features
        nan_cols = np.isnan(_x).any(axis=0) | np.isnan(_y).any(axis=0)
        if nan_cols.any():
            warnings.warn('NaN column removed ({})'.format(np.sum(nan_cols)))
            _x = _x[:, ~nan_cols]
            _y = _y[:, ~nan_cols]

    r = np.corrcoef( _x, _y)[:n_sample, n_sample:]

    return r


def pairwise_identification(pred, true, metric='correlation', remove_nan=True, remove_nan_dist=True, single_trial=False, pred_labels=None, true_labels=None):
    """Pair-wise identification."""
    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)

    if remove_nan:
        # Remove nan columns based on the decoded features
        nan_cols = np.isnan(p).any(axis=0) | np.isnan(t).any(axis=0)
        if nan_cols.any():
            warnings.warn('NaN column removed ({})'.format(np.sum(nan_cols)))
            p = p[:, ~nan_cols]
            t = t[:, ~nan_cols]

    if single_trial:
        cr = []
        for i in range(p.shape[0]):
            d = 1 - cdist(p[i][np.newaxis], t, metric=metric)
            # label の情報
            ind = np.where(np.array(true_labels) == pred_labels[i])[0][0]

            s = (d - d[0, ind]).ravel()
            if remove_nan_dist and np.isnan(s).any():
                warnings.warn('NaN value detected in the distance matrix ({}).'.format(np.sum(np.isnan(s))))
                s = s[~np.isnan(s)]
            ac = np.sum(s < 0) / (len(s) - 1)
            cr.append(ac)
        cr = np.asarray(cr)
    else:
        if metric == 'correlation':
            d = _correlation_similarity(p, t)
        else:
            d = 1 - cdist(p, t, metric=metric)

        if remove_nan_dist:
            cr = []
            for d_ind in range(d.shape[0]):
                pef = d[d_ind, :] - d[d_ind, d_ind]
                if np.isnan(pef).any():
                    warnings.warn('NaN value detected in the distance matrix ({}).'.format(np.sum(np.isnan(pef))))
                    pef = pef[~np.isnan(pef)] # Remove nan value from the comparison for identification
                pef = np.sum(pef < 0) / (len(pef) - 1)
                cr.append(pef)
            cr = np.asarray(cr)
        else:
            cr = np.sum(d - np.diag(d)[:, np.newaxis] < 0, axis=1) / (d.shape[1] - 1)

    return cr


def remove_nan_value(array, nan_flag=None, return_nan_flag=False):
    """Remove columns (units) which contain NaN values.

    Parameters
    ----------
    array : numpy.ndarray
        Input array of shape (n_samples, n_units).
    nan_flag : numpy.ndarray or list, optional
        Boolean mask of columns to remove. If not given, it is computed from
        ``array``.
    return_nan_flag : bool, optional
        If True, also return the NaN flag used for removal (default: False).

    Returns
    -------
    nan_removed_array : numpy.ndarray
        Array with NaN-containing columns removed.
    nan_flag : numpy.ndarray
        Boolean mask used for removal. Only returned when
        ``return_nan_flag=True``.
    """
    if nan_flag is None:
        nan_flag = np.isnan(array).any(axis=0)
    nan_removed_array = array[:, ~nan_flag]

    if return_nan_flag:
        return nan_removed_array, nan_flag
    else:
        return nan_removed_array
