'''Utility functions for BData'''


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

    dat = bdpy.BData()

    for ds in data_list:
        for s in successive:
            v = ds.get_dataset(s)
            v += suc_cols[s]
            ds.update(s, v)

        if dat.dataSet.shape[0] == 0:
            dat.dataSet = ds.dataSet
            dat.metaData = ds.metaData
        else:
            dat.dataSet = np.vstack([dat.dataSet, ds.dataSet])
            # TODO: add metadat check

        for s in successive:
            v = dat.get_dataset(s)
            suc_cols[s] = np.max(v)

    return dat
