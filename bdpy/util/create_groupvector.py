"""
create_groupvector
"""


__all__ = ['create_groupvector']


import numpy as np


def create_groupvector(group_label, group_size):
    """Create a group vector

    Parameters
    ----------
    group_label : array_like
        List or array of group labels
    group_size : array_like
        Sample size of each group

    Returns
    -------
    group_vector : array
        A vector specifying groups (size: 1 * N)

    Example:

        >>> bdpy.util.create_groupvector([ 1, 2, 3 ], 2)
        array([1, 1, 2, 2, 3, 3])

        >>> bdpy.util.create_groupvector([ 1, 2, 3 ], [ 2, 4, 2 ])
        array([1, 1, 2, 2, 2, 2, 3, 3])
    """
    
    group_vector = []

    if isinstance(group_size, int):
        # When 'group_size' is integer, create array in which each group label
        # is repeated for 'group_size'
        group_size_list = [ group_size for _ in xrange(len(group_label)) ]
    elif isinstance(group_size, list) | isinstance(group_size, np.ndarray):
        if len(group_label) != len(group_size):
            raise ValueError("Length of 'group_label' and 'group_size' is mismatched")
        group_size_list = group_size
    else:
        raise TypeError("Invalid type of 'group_size'")

    group_list = [np.array([label for _ in range(group_size_list[i])]) for i, label in enumerate(group_label)]
    group_vector = np.hstack(group_list)

    return group_vector
