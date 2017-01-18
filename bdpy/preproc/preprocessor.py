"""
Classes for preprocessing

This file is a part of BdPy.
"""


import sys
from abc import ABCMeta, abstractmethod

import numpy as np


## Abstract preprocessor #######################################################

class Preprocessor(object):
    """
    Abstract class for preprocessing
    """

    __metaclass__ = ABCMeta
    
    @abstractmethod
    def proc(self, x, ind, opt):
        """
        Abstract method of preprocessing

        `proc` should return `y` and `ind_map`
        """
        pass

    def run(self, x, group, **kargs):
        """
        Template method of preprocessing
        """

        # If `group` is empty, apply preprocessing to the whole data
        if len(group) == 0:
            group = np.ones(x.shape[0])
        
        group = np.array(group) # Input `group` can be either np.array or list

        group_set = sorted(list(set(group)))

        prec_data = []
        ind_maps = []
        
        for g in group_set:
            group_index = np.where(group == g)[0]
            group_data = x[group_index]

            pdata, pind = self.proc(group_data, group_index, kargs)

            prec_data.append(pdata)
            ind_maps.append(pind)

        y = np.vstack(prec_data)
        index_map = np.hstack(ind_maps) # `index_map` should be a vector
        
        return y, index_map


## Preprocessor classes ########################################################

class Average(Preprocessor):

    def proc(self, x, ind, opt):

        x_ave = np.average(x, axis = 0)

        ind_map = ind[0]
        
        return x_ave, ind_map


class Detrender(Preprocessor):

    def proc(self, x, ind, opt):

        from scipy.signal import detrend

        keep_mean = opt['keep_mean']
        
        x_mean = np.mean(x, axis = 0)
        x_detl = detrend(x, axis = 0, type = 'linear')

        if keep_mean:
            x_detl += x_mean

        ind_map = ind
        
        return x_detl, ind_map


class Normalize(Preprocessor):

    def proc(self, x, ind, opt):

        mode = opt['mode']
        
        x_mean = np.mean(x, axis = 0)
        x_std = np.std(x, axis = 0)

        if mode == "PercentSignalChange":
            # zero division outputs nan
            # np.nan_to_num convert nan to 0.
            # Is this correct? Should cells divided by zero return inf?
            x = np.nan_to_num(np.divide(100.0 * (x - x_mean), x_mean))
        elif mode == "Zscore":
            x = np.nan_to_num(np.divide((x - x_mean), x_std))
        elif mode == "DivideMean":
            x = np.nan_to_num(np.divide(100.0 * x, x_mean))
        elif mode == "SubtractMean":
            x = x - x_mean
        else:
            NameError("Unknown normalization mode: %s', norm_mode" % (mode))

        ind_map = ind
            
        return x, ind_map


class ShiftSample(Preprocessor):

    def proc(self, x, ind, opt):

        s = opt['shift_size']
        
        y = x[s:]
        ind_map = ind[:-s]

        return y, ind_map
