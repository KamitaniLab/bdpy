"""Utility functions for BData."""


from typing import List

import copy

import numpy as np

from .bdata import BData


def vstack(bdata_list, successive=[], metadata_merge='strict', ignore_metadata_description=False):
    """Concatenate datasets vertically.

    Currently, `concat_dataset` does not validate the consistency of meta-data
    among data.

    Parameters
    ----------
    bdata_list : list of BData
        Data to be concatenated
    successsive : list, optional
        Sucessive columns. The values of columns specified here are inherited
        from the preceding data.
    metadata_merge : str, optional
        Meta-data merge strategy ('strict' or 'minimal'; default: strict).
        'strict' requires that concatenated datasets share exactly the same meta-data.
        'minimal' keeps meta-data only shared across the concatenated datasets.
    ignore_metadata_description : bool
        Ignore meta-data description when merging the datasets (default: False).

    Returns
    -------
    dat : BData
        Concatenated data

    Example
    -------

        data = vstack([data0, data1, data2], successive=['Session', 'Run', 'Block'])
    """

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

            # vmap
            vmap_keys = ds_copy.get_vmap_keys()
            for vk in vmap_keys:
                dat.add_vmap(vk, ds_copy.get_vmap(vk))
        else:
            # Check metadata consistency
            if metadata_merge == 'strict':
                if not metadata_equal(dat, ds_copy):
                    raise ValueError('Inconsistent meta-data.')
            elif metadata_merge == 'minimal':
                # Only meta-data shared across BDatas are kept.
                shared_mkeys = sorted(list(set(dat.metadata.key) & set(ds_copy.metadata.key)))
                shared_mdesc = []
                shared_mvalue_lst = []
                for mkey in shared_mkeys:
                    d0_desc, d0_value = dat.metadata.get(mkey, 'description'), dat.metadata.get(mkey, 'value')
                    d1_desc, d1_value = ds_copy.metadata.get(mkey, 'description'), ds_copy.metadata.get(mkey, 'value')

                    if not ignore_metadata_description and not d0_desc == d1_desc:
                        raise ValueError('Inconsistent meta-data description (%s)' % mkey)
                    try:
                        np.testing.assert_equal(d0_value, d1_value)
                    except AssertionError:
                        raise ValueError('Inconsistent meta-data value (%s)' % mkey)
                    shared_mdesc.append(d0_desc)
                    shared_mvalue_lst.append(d0_value)
                shared_mvalue = np.vstack(shared_mvalue_lst)

                dat.metadata.key = shared_mkeys
                dat.metadata.description = shared_mdesc
                dat.metadata.value = shared_mvalue
            else:
                raise ValueError('Unknown meta-data merge strategy: %s' % metadata_merge)

            # Concatenate BDatas
            dat.dataset = np.vstack([dat.dataset, ds_copy.dataset])

            # Merge vmap
            vmap_keys = ds_copy.get_vmap_keys()
            for vk in vmap_keys:
                dat.add_vmap(vk, ds_copy.get_vmap(vk))

        # Update the last values in sucessive columns
        for s in successive:
            v = dat.select(s)
            suc_cols[s] = np.max(v)

    return dat


def resolve_vmap(bdata_list):
    """Replace the conflicting vmaps for multiple bdata with non-conflicting vmaps.

    Parameters
    ----------
    bdata_list : list of BData
        Data to be concatenated

    Returns
    -------
    bdata_list : list of BData
        The vmap is fixed to avoid a collision.
    """
    # Get the vmap key list.
    vmap_keys = bdata_list[0].get_vmap_keys()

    # Check each vmap key.
    for vmap_key in vmap_keys:
        new_vmap = {}
        # Check each bdata vmap.
        for ds in bdata_list:
            vmap = ds.get_vmap(vmap_key)
            ds_values, selector = ds.select(vmap_key, return_index = True) # keep original dataset values
            new_dsvalues = copy.deepcopy(ds_values)  # to update

            # Sanity check
            if not vmap_key in ds.metadata.key:
                raise ValueError('%s not found in metadata.' % vmap_key)
            if type(vmap) is not dict:
                raise TypeError('`vmap` should be a dictionary.')
            for vk in vmap.keys():
                if type(vk) is str:
                    raise TypeError('Keys of `vmap` should be numerical.')

            # Check duplicate and create new vmap
            for vk in vmap.keys():
                if vk not in new_vmap.keys():
                    # If find a novel key, add the new key and new value
                    new_vmap[vk] = vmap[vk]
                elif new_vmap[vk] != vmap[vk]:
                    # If find the exisiting key and the values are different,
                    # assign a new key by incrementing 1 to the maximum exisiting key.
                    inflation_key_value = max(new_vmap.keys())
                    new_vmap[inflation_key_value + 1] = vmap[vk]
                    # Update dataset values
                    new_dsvalues[ds_values == vk] = inflation_key_value + 1
                else:
                    # If the key and value is same, nothing to do.
                    pass

            # Update dataset
            ds.dataset[:, selector] = new_dsvalues

        # Update each bdata vmap.
        for ds in bdata_list:
            vmap = ds.get_vmap(vmap_key)
            if not np.array_equal(sorted(list(vmap.keys())), sorted(list(new_vmap.keys()))):
                # If the present vmap is different from new_vmap, update it.
                ds._BData__vmap[vmap_key] = new_vmap # BDataクラスにvmapのsetterがあると良い

    return bdata_list


def concat_dataset(data_list, successive=[]):
    """Concatenate datasets

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
    """

    return vstack(data_list, successive=successive)


def metadata_equal(d0, d1, strict=False):
    """Check whether `d0` and `d1` share the same meta-data.

    Parameters
    ----------
    d0, d1 : BData
    strict : bool, optional

    Returns
    -------
    bool
    """

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
            np.testing.assert_equal(d0_mval, d1_mval)
        except AssertionError:
            return False

    return True


def select_data_multi_bdatas(bdatas: List[BData], roi_selector: str) -> np.ndarray:
    """Load brain data from multiple BDatas."""
    return np.vstack([b.select(roi_selector) for b in bdatas])


def get_labels_multi_bdatas(bdatas: List[BData], label_name: str) -> List[str]:
    """Load brain data from multiple BDatas."""
    return [label for b in bdatas for label in b.get_labels(label_name)]
