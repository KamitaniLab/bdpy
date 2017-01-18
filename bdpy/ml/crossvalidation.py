"""
crossvalidation
"""


import numpy as np


def make_cvindex(group):
    """
    Make indexes of training and test samples

    Args:

    - group : Index partitioning samples to groups (N x 1 array; N is the total
              number of samples). If 'group' is runs_groups -> crossvalidation
              between run. If 'group' is block_groups -> leave-on-out
              crossvalidation. If all elements in 'group' are 0 -> training
              only. If all elements in 'group' are 1 -> test only.

    Returns:

    - train_index : Boolean matrix specifying training samples (N x K array; N
                    is the total number of samples, K is the number of folds)
    - test_index : Boolean matrix specifying test samples (N x K array)

    Example:

        >>> bdpy.util.make_crossvalidationindex(np.array([1, 1, 2, 2, 3, 3]))
        (array([[False,  True,  True],
                [False,  True,  True],
                [ True, False,  True],
                [ True, False,  True],
                [ True,  True, False],
                [ True,  True, False]], dtype=bool),
         array([[ True, False, False],
                [ True, False, False],
                [False,  True, False],
                [False,  True, False],
                [False, False,  True],
                [False, False,  True]], dtype=bool))
    """
    
    # Get and sort unique group index
    group_set = sorted(list(set(group.flatten())))
    
    # The number of sample 
    n_sample = len(group);
    # The number of run
    n_group = len(group_set);
    
    if n_group == 0:
        # Training only
        train_index = np.ones((n_sample,))
        test_index = np.zeros((n_sample,))
    elif n_group == 1:
        # Test only
        train_index = np.zeros((n_sample,))
        test_index = np.ones((n_sample,))
    else:
        # K-fold crossvalidation
        group_t = np.reshape(group, (group.shape[0], 1))
        mat_group = np.tile(group_t, (1, n_group))
        mat_group_set = np.tile(np.array(group_set), (n_sample, 1))
        
        train_index = mat_group != mat_group_set
        test_index = mat_group == mat_group_set
    
    return train_index, test_index


def make_crossvalidationindex(group):
    """
    Make indexes of training and test samples

    Args:

    - group : Index partitioning samples to groups (N x 1 array; N is the total
              number of samples). If 'group' is runs_groups -> crossvalidation
              between run. If 'group' is block_groups -> leave-on-out
              crossvalidation. If all elements in 'group' are 0 -> training
              only. If all elements in 'group' are 1 -> test only.

    Returns:

    - train_index : Boolean matrix specifying training samples (N x K array; N
                    is the total number of samples, K is the number of folds)
    - test_index : Boolean matrix specifying test samples (N x K array)

    Example:

        >>> bdpy.util.make_crossvalidationindex(np.array([1, 1, 2, 2, 3, 3]))
        (array([[False,  True,  True],
                [False,  True,  True],
                [ True, False,  True],
                [ True, False,  True],
                [ True,  True, False],
                [ True,  True, False]], dtype=bool),
         array([[ True, False, False],
                [ True, False, False],
                [False,  True, False],
                [False,  True, False],
                [False, False,  True],
                [False, False,  True]], dtype=bool))
    """

    return make_cvindex(group)
