'''Utility functions for BData'''

import copy

import numpy as np

from .bdata import BData


def vstack(bdata_list, successive=[]):
    '''Concatenate datasets vertically.

    Currently, `concat_dataset` does not validate the consistency of meta-data
    among data.

    Parameters
    ----------
    bdata_list : list of BData
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

        data = vstack([data0, data1, data2], successive=['Session', 'Run', 'Block'])
    '''

    suc_cols = {s : 0 for s in successive}

    dat = BData()  # Concatenated BData

    for ds in bdata_list:
        ds_copy = copy.deepcopy(ds)

        # Update sucessive columns
        for s in successive:
            v = ds_copy.select(s)
            v += suc_cols[s]
            ds_copy.update(s, v)

        # Concatenate BDatas
        if dat.dataset.shape[0] == 0:
            # Create new BData
            dat.dataset = ds_copy.dataset
            dat.metadata = ds_copy.metadata
        else:
            # Concatenate BDatas
            dat.dataset = np.vstack([dat.dataset, ds_copy.dataset])

            # Check metadata consistency
            if not dat.metadata.key == ds_copy.metadata.key:
                raise ValueError('Metadata keys are inconsistent. ')
            if not dat.metadata.description == ds_copy.metadata.description:
                raise ValueError('Metadata descriptions are inconsistent. ')
            # np.array_equal doesn't work because np.nan != np.nan
            try:
                np.testing.assert_equal(dat.metadata.value, ds_copy.metadata.value)
            except AssertionError:
                raise ValueError('Metadata values are inconsistent. ')

        # Update the last values in sucessive columns
        for s in successive:
            v = dat.select(s)
            suc_cols[s] = np.max(v)

    return dat


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

    return vstack(data_list, successive=successive)


def metadata_equal(d0, d1, strict=False):
    '''Check whether `d0` and `d1` share the same meta-data.'''

    equal = True

    # Strict check
    if not d0.metadata.key == d1.metadata.key:
        equal = False
    if not d0.metadata.description == d1.metadata.description:
        equal = False
    try:
        np.testing.assert_equal(d0.metadata.value, d1.metadata.value)
    except AssertionError:
        equal = False

    if equal:
        return True

    if strict:
        return False

    # Loose check (ignore the order of meta-data)
    d0_mkeys = sorted(d0.metadata.key)
    d1_mkeys = sorted(d1.metadata.key)
    if not d0_mkeys == d1_mkeys:
        return False

    for mkey in d0_mkeys:
        d0_mdesc, d1_mdesc = d0.metadata.get(mkey, 'description'), d1.metadata.get(mkey, 'description')
        d0_mval, d1_mval = d0.metadata.get(mkey, 'value'), d1.metadata.get(mkey, 'value')

        if not d0_mdesc == d1_mdesc:
            return False

        try:
            np.testing.assert_equal(d0.metadata.value, d1.metadata.value)
        except AssertionError:
            return False

    return True
