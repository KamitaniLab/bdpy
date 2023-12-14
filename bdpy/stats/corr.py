"""Functions dealing with correlation.

This file is a part of BdPy.
"""

__all__ = ['corrcoef', 'corrmat']


import numpy as np
from numpy.matlib import repmat


def corrcoef(x: np.ndarray, y: np.ndarray, var: str = 'row') -> np.ndarray:
    """Return correlation coefficients between `x` and `y`.

    Parameters
    ----------
    x, y : array_like
        Matrix or vector
    var : str, 'row' or 'col'
        Specifying whether rows (default) or columns represent variables

    Returns
    -------
    r
        Correlation coefficients
    """
    # Convert vectors to arrays
    if x.ndim == 1:
        x = np.array([x])

    if y.ndim == 1:
        y = np.array([y])

    # Normalize x and y to row-var format
    if var == 'row':
        # 'rowvar=1' in np.corrcoef

        # Vertical vector --> horizontal
        if x.shape[1] == 1:
            x = x.T
        if y.shape[1] == 1:
            y = y.T
    elif var == 'col':
        # 'rowvar=0' in np.corrcoef

        # Horizontal vector --> vertical
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T

        # Convert to rowvar=1
        x = x.T
        y = y.T
    else:
        raise ValueError('Unknown var parameter specified')

    # Match size of x and y
    if x.shape[0] == 1 and y.shape[0] != 1:
        x = repmat(x, y.shape[0], 1)
        
    elif x.shape[0] != 1 and y.shape[0] == 1:
        y = repmat(y, x.shape[0], 1)

    # Check size of normalized x and y
    if x.shape != y.shape:
        raise TypeError('Input matrixes size mismatch')

    # Get num variables
    nvar = x.shape[0]

    # Get correlation
    rmat = np.corrcoef(x, y, rowvar=1)
    r = np.diag(rmat[:nvar, nvar:])

    return r


def corrmat(x: np.ndarray, y: np.ndarray, var: str = 'row') -> np.ndarray:
    """Return correlation matrix between `x` and `y`.

    Parameters
    ----------
    x, y : array_like
        Matrix or vector
    var : str, 'row' or 'col'
        Specifying whether rows (default) or columns represent variables

    Returns
    -------
    rmat
        Correlation matrix
    """
    # Fix x and y to represent variables in each row
    if var == 'row':
        pass
    elif var == 'col':
        x = x.T
        y = y.T
    else:
        raise ValueError('Unknown var parameter specified')

    nobs = x.shape[1]

    # Subtract mean(a, axis=1) from a
    def submean(a: np.ndarray) -> np.ndarray:
        return a - np.matrix(np.mean(a, axis=1)).T

    cmat = (np.dot(submean(x), submean(y).T) / (nobs - 1)) / np.dot(np.matrix(np.std(x, axis=1, ddof=1)).T, np.matrix(np.std(y, axis=1, ddof=1)))
    
    return np.array(cmat)
