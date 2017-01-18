"""
Interface functions for preprocessing

This file is a part of BdPy
"""


from preprocessor import Average,Detrender,Normalize,ShiftSample
from util import print_start_msg, print_finish_msg


def average_sample(x, group, verbose = True):
    """
    Average samples within groups

    Parameters
    ----------
    x     : input data array (size: sample num * feature num)
    group : group vector (length: sample num)

    Returns
    -------
    y         : averaged data array (size: group num * feature num)
    index_map : vector mapping row indexes from y to x (length: group num)
    """

    if verbose:
        print_start_msg()

    p = Average()
    y, ind_map = p.run(x, group)
    
    if verbose:
        print_finish_msg()

    return y, ind_map


def detrend_sample(x, group, keep_mean=True, verbose=True):
    """
    Apply linear detrend

    Parameters
    ----------
    x     : input data array (size: sample num * feature num)
    group : group vector (length: sample num)
    
    Returns
    -------
    y : detrended data array (size: sample num * feature num)
    """

    if verbose:
        print_start_msg()

    p = Detrender()
    y, _ = p.run(x, group, keep_mean=keep_mean)
        
    if verbose:
        print_finish_msg()

    return y


def normalize_sample(x, group,
                     mode = 'PercentSignalChange',
                     baseline = 'All',
                     zero_threshold = 1,
                     verbose = True):
    """
    Apply normalization

    Parameters
    ----------
    x             : input data array (size: sample num * feature num)
    group         : group vector (length: sample num)
    Mode          : normalization mode ('PercentSignalChange', 'Zscore',
                    'DivideMean', or 'SubtractMean';
                    default = 'PercentSignalChange')
    Baseline      : baseline index vector (default: 'allsamples')
    ZeroThreshold : zero threshold (default: 1)

    Returns
    -------
    y : normalized data array (size: sample num * feature num)
    """

    if verbose:
        print_start_msg()

    p = Normalize()
    y, _ = p.run(x, group, mode = mode, baseline = baseline, zero_threshold = zero_threshold)

    if verbose:
        print_finish_msg()

    return y


def shift_sample(x, group, shift_size = 1, verbose = True):
    """
    Shift sample within groups

    Parameters
    ----------
    x          : input data array (size: sample num * feature num)
    group      : group vector (length: sample num)
    shift_size : shift size (default: 1)  

    Returns
    -------
    y         : averaged data array (size: group num * feature num)
    index_map : vector mapping row indexes from y to x (length: group num)

    Example:

    import numpy as np
    from bdpy.preprocessor import shift_sample

    x = np.array([[  1,  2,  3 ],
                  [ 11, 12, 13 ],
                  [ 21, 22, 23 ],
                  [ 31, 32, 33 ],
                  [ 41, 42, 43 ],
                  [ 51, 52, 53 ]])
    grp = np.array([ 1, 1, 1, 2, 2, 2 ])
        
    shift_size = 1

    y, index = shift_sample(x, grp, shift_size)

    # >>> y
    # array([[11, 12, 13],
    #        [21, 22, 23],
    #        [41, 42, 43],
    #        [51, 52, 53]])

    # >>> index
    # array([0, 1, 3, 4])
    """

    if verbose:
        print_start_msg()

    p = ShiftSample()
    y, index_map = p.run(x, group, shift_size = shift_size)

    if verbose:
        print_finish_msg()

    return y, index_map
