"""select_top.

This file is a part of BdPy.
"""


__all__ = ['select_top']


from typing import Tuple, Optional
import numpy as np
from .util import print_start_msg, print_finish_msg


def select_top(data: np.ndarray, value: np.ndarray, num: int, axis: Optional[int] = 0, verbose: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
    """Select top `num` features of `value` from `data`.

    Parameters
    ----------
    data : array
       Data matrix
    value : array_like
       Vector of values
    num : int
       Number of selected features

    Returns
    -------
    selected_data : array
        Selected data matrix
    selected_index : array
        Index of selected data

    """
    if verbose:
        print_start_msg()

    num_elem = data.shape[axis]

    value = np.array([-np.inf if np.isnan(a) else a for a in value])
    sorted_index = np.argsort(value)[::-1]

    rank = np.zeros(num_elem, dtype=int)
    rank[sorted_index] = np.array(range(0, num_elem))

    selected_index_bool = rank < num

    if axis == 0:
        selected_data = data[selected_index_bool, :]
        selected_index = np.array(range(0, num_elem), dtype=int)[selected_index_bool]
    elif axis == 1:
        selected_data = data[:, selected_index_bool]
        selected_index = np.array(range(0, num_elem), dtype=int)[selected_index_bool]
    else:
        raise ValueError('Invalid axis')

    if verbose:
        print_finish_msg()

    return selected_data, selected_index
