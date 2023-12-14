"""Math utils."""


from typing import List

import numpy as np


def average_elemwise(arrays: List[np.ndarray], keepdims: bool = False) -> np.ndarray:
    """Return element-wise mean of arrays.

    Parameters
    ----------
    arrays : list of ndarrays
      List of arrays.
    keepdims : bool
      Keep dimension in returned array or not.

    Return
    ------
    ndarray
    """
    n_array = len(arrays)

    max_dim_i = np.argmax([a.ndim for a in arrays])
    max_array_shape = arrays[max_dim_i].shape

    arrays_sum = np.zeros(max_array_shape)

    for a in arrays:
        arrays_sum += a

    mean_array = arrays_sum / n_array

    if not keepdims:
        mean_array = np.squeeze(mean_array)
        
    return mean_array
