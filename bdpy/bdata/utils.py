'''Utility functions for BData'''

import copy

import numpy as np

from bdata import BData


def concat_dataset(data_list, successive=[]):
    '''Concatenate datasets

    Currently, `concat_dataset` does not validate the consistency of meta-data
    among data.

    Parameters
    ----------
    data_list : list of BData
        Data to be concatenated
    successsive : list, optional
        Sucessive columns. The values of columns specified here are inherited
        from the preceding data.
    
    Returns
    -------
    dat : BData
        Concatenated data

    Example
    -------
    
        data = concat_dataset([data0, data1, data2], successive=['Session', 'Run', 'Block'])
    '''

    suc_cols = {s : 0 for s in successive}

    dat = BData()

    for ds in data_list:
        ds_copy = copy.deepcopy(ds)
        for s in successive:
            v = ds_copy.select(s)
            v += suc_cols[s]
            ds_copy.update(s, v)

        if dat.dataset.shape[0] == 0:
            dat.dataset = ds_copy.dataset
            dat.metadata = ds_copy.metadata
        else:
            dat.dataset = np.vstack([dat.dataset, ds_copy.dataset])
            # TODO: add metadat check

        for s in successive:
            v = dat.select(s)
            suc_cols[s] = np.max(v)

    return dat
