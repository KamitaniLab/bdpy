"""
BrainDecoderToolbox2 data class

This file is a part of BdPy


API lits
--------

- Modify data
    - add_dataset
    - add_metadata
    - rename_meatadata
    - set_metadatadescription
- Search and extract data
    - select_dataset
    - get_metadata
- Get information
    - show_metadata
- File I/O
    - load
    - save
"""


__all__ = ['BData']


import os

import h5py
import numpy as np
import scipy.io as sio

from metadata import MetaData
from featureselector import FeatureSelector


class BData(object):
    """
    Class for BdPy/BrainDecoderToolbox2 data (dataSet and metaData)

    The instance of class `BData` contains `dataSet` and `metaData` as instance
    variables.

    Attributes
    ----------
    dataSet : numpy array (dtype=float)
        Dataset array
    metaData : MetaData object
        Meta-data object
    """


    def __init__(self, file_name=None, file_type=None):
        """
        Initializer of a BData instance

        Parameters
        ----------
        file_name : str, optional
            File which contains BData (default: None)
        file_type : {'Npy', 'Matlab', 'HDF5'}
            File type (default: None)
        """

        self.dataSet = np.ndarray((0, 0), dtype=float)
        self.metaData = MetaData()

        if file_name is not None:
            self.load(file_name, file_type)


    ## Public APIs #####################################################

    def add_dataset(self, x, attribute_key):
        """
        Add `x` to dataSet with attribute meta-data key `attribute_key`

        Parameters
        ----------
        x : array
            Data matrix to be added in dataSet
        attribute_key : str
            Key of attribute meta-data, which specifies the columns containing `x`
        """

        if x.ndim == 1:
            x = x[:, np.newaxis]

        colnum_has = self.dataSet.shape[1] # Num of existing columns in 'dataSet'
        colnum_add = x.shape[1]            # Num of columns to be added

        ## Add 'x' to dataset
        if not self.dataSet.size:
            self.dataSet = x
        else:
            # TODO: Add size check of 'x' and 'self.dataSet'
            self.dataSet = np.hstack((self.dataSet, x))

        ## Add new attribute metadata
        attribute_description = 'Attribute: %s = 1' % attribute_key
        attribute_value = [None for _ in xrange(colnum_has)] + [1 for _ in xrange(colnum_add)]
        self.metaData.set(attribute_key, attribute_value, attribute_description,
                          lambda x, y: np.hstack((y[:colnum_has], x[-colnum_add:])))


    def add_metadata(self, key, value, description='', attribute=None):
        """
        Add meta-data with `key`, `description`, and `value` to metaData

        Parameters
        ----------
        key : str
            Meta-data key
        value : array
            Meta-data array
        description : str, optional
            Meta-data description
        attribute : str, optional
            Meta-data key specifying data attribution
        """

        # TODO: Add attribute specifying
        # TODO: Add size check of metaData/value

        if attribute is not None:
            attr_ind = np.asarray(np.nan_to_num(self.metaData.get(attribute, 'value')), dtype=np.bool)
            # nan is converted to True in np.bool and thus change nans in metaData/value to zero at first

            add_value = np.array([None for _ in xrange(self.metaData.get_value_len())])
            add_value[attr_ind] = value
        else:
            add_value = value

        self.metaData.set(key, add_value, description)


    def rename_meatadata(self, key_old, key_new):
        """
        Rename a meta-data key

        Parameters
        ----------
        key_old, key_new : str
            Old and new meta-data keys
        """
        self.metaData[key_new] = self.metaData[key_old]
        del self.metaData[key_old]


    def set_metadatadescription(self, key, description):
        """
        Set description of metadata specified by `key`

        Parameters
        ----------
        key : str
            Meta-data key
        description : str
            Meta-data description
        """

        self.metaData.set(key, None, description,
                          lambda x, y: y)


    def select_dataset(self, condition, return_index=False, verbose=True):
        """
        Extracts features from dataset based on condition

        Parameters
        ----------
        condition : str
            Expression specifying feature selection
        retrun_index : bool, optional
            If True, returns index of selected features (default: False)
        verbose : bool, optional
            If True, display verbose messages (default: True)

        Returns
        -------
        array
            Selected feature data and index (if specified)
        list, optional
            Selected index

        Note
        ----

        Operators: | (or), & (and), = (equal), @ (conditional)
        """

        expr_rpn = FeatureSelector(condition).rpn

        stack = []
        buf_sel = []

        for i in expr_rpn:
            if i == '=':
                r = stack.pop()
                l = stack.pop()

                stack.append(np.array([n == r for n in l], dtype=bool))

            elif i == 'top':
                # Dirty solution

                # Need fix on handling 'None'

                n = int(stack.pop()) # Num of elements to be selected
                v = stack.pop()

                order = self.__get_order(v)

                stack.append(order)
                buf_sel.append(n)

            elif i == '|' or i == '&':
                r = stack.pop()
                l = stack.pop()

                if r.dtype != 'bool':
                    # 'r' should be an order vector
                    num_sel = buf_sel.pop()
                    r = self.__get_top_elm_from_order(r, num_sel)
                    #r = np.array([ n < num_sel for n in r ], dtype = bool)

                if l.dtype != 'bool':
                    # 'l' should be an order vector
                    num_sel = buf_sel.pop()
                    l = self.__get_top_elm_from_order(l, num_sel)
                    #l = np.array([ n < num_sel for n in l ], dtype = bool)

                if i == '|':
                    result = np.logical_or(l, r)
                elif i == '&':
                    result = np.logical_and(l, r)

                stack.append(result)

            elif i == '@':
                # FIXME
                # In the current version, the right term of '@' is assumed to
                # be a boolean, and the left is to be an order vector.

                r = stack.pop() # Boolean
                l = stack.pop() # Float

                l[~r] = np.inf

                selind = self.__get_top_elm_from_order(l, buf_sel.pop())

                stack.append(np.array(selind))

            else:
                if isinstance(i, str):
                    if i.isdigit():
                        # 'i' should be a criteria value
                        i = float(i)
                    else:
                        # 'i' should be a meta-data key
                        i = np.array(self.get_metadata(i))

                stack.append(i)

        selected_index = stack.pop()

        # If buf_sel still has an element, `select_index` should be an order vector.
        # Select N elements based on the order vector.
        if buf_sel:
            num_sel = buf_sel.pop()
            selected_index = [n < num_sel for n in selected_index]

        # get whole dataset
        #data = self.get_dataset()

        # slice dataset based on selected column
        #feature = data[:, np.array(selected_index)]

        if return_index:
            return self.dataSet[:, np.array(selected_index)], selected_index
        else:
            return self.dataSet[:, np.array(selected_index)]


    def get_metadata(self, key):
        """
        Get value of meta-data specified by 'key'
        """

        return self.metaData.get(key, 'value')


    def show_metadata(self):
        """
        Show all the key and description in metaData
        """

        for m in self.metaData:
            print "%s: %s" % (m['key'], m['description'])


    def load(self, load_filename, load_type=None):
        """
        Load 'dataSet' and 'metaData' from a given file
        """

        if load_type is None:
            load_type = self.__get_filetype(load_filename)

        if load_type == "Npy":
            self.__load_npy(load_filename)
        elif load_type == "Matlab":
            self.__load_mat(load_filename)
        elif load_type == "HDF5":
            self.__load_h5(load_filename)
        else:
            raise ValueError("Unknown file type: %s" % (load_type))


    def save(self, file_name, file_type=None):
        """
        Save 'dataSet' and 'metaData' to a file
        """

        if file_type is None:
            file_type = self.__get_filetype(file_name)

        if file_type == "Npy":
            np.save(file_name, {"dataSet": self.dataSet,
                                "metaData": self.metaData})
        elif file_type == "Matlab":
            md_key = []
            md_desc = []
            md_value = []

            for m in self.metaData:
                md_key.append(m['key'])
                md_desc.append(m['description'])

                v_org = m['value']
                v_nan = []

                # Convert 'None' to 'np.nan'
                for v in v_org:
                    if v is None:
                        v_nan.append(np.nan)
                    else:
                        v_nan.append(v)

                md_value.append(v_nan)

            # 'key' and 'description' are saved as cell arrays
            sio.savemat(file_name, {"dataSet" : self.dataSet,
                                    "metaData" : {"key" : np.array(md_key, dtype=np.object),
                                                  "description" : np.array(md_desc, dtype=np.object),
                                                  "value" : md_value}})

        elif file_type == "HDF5":
            self.__save_h5(file_name)

        else:
            raise ValueError("Unknown file type: %s" % (file_type))


    ## Public APIs (obsoleted) #########################################

    def get_dataset(self, key=None):
        """
        Get dataSet from BData object

        When `key` is not given, `get_dataset` returns `dataSet`. When `key` is
        given, `get_dataset` returns data specified by `key`
        """

        if key is None:
            return self.dataSet
        else:
            query = '%s = 1' % key
            return self.select_dataset(query, return_index=False, verbose=False)


    ## Feature selection #######################################################

    def edit_metadatadescription(self, metakey, description):
        """
        Add or edit description of metadata based on key

        This method is obsoleted and will be removed in the future release.
        Use `set_metadatadescription` instead.

        Parameters
        ----------
        key : str
            Meta-data key
        description : str
            Meta-data description
        """
        self.set_metadatadescription(metakey, description)


    def select_feature(self, condition, return_index=False, verbose=True):
        """
        Extracts features from dataset based on condition

        Parameters
        ----------
        condition : str
            Expression specifying feature selection
        retrun_index : bool, optional
            If True, returns index of selected features (default: False)
        verbose : bool, optional
            If True, display verbose messages (default: True)

        Returns
        -------
        array
            Selected feature data and index (if specified)
        list, optional
            Selected index

        Note
        ----

        Operators: | (or), & (and), = (equal), @ (conditional)
        """
        return self.select_dataset(condition, return_index, verbose)


    ## Private methods #################################################

    def __get_order(self, v, sort_order='descend'):

        # 'np.nan' comes to the last of an acending series, and thus the top of a decending series.
        # To avoid that, convert 'np.nan' to -Inf.
        v[np.isnan(v)] = -np.inf

        sorted_index = np.argsort(v)[::-1] # Decending order
        order = range(len(v))
        for i, x in enumerate(sorted_index):
            order[x] = i

        return np.array(order, dtype=float)


    def __get_top_elm_from_order(self, order, n):
        """Get a boolean index of top 'n' elements from 'order'"""
        sorted_index = np.argsort(order)
        for i, x in enumerate(sorted_index):
            order[x] = i

        index = np.array([r < n for r in order], dtype=bool)

        return index


    def __save_h5(self, file_name):
        """
        Save data in HDF5 format (*.h5)
        """

        with h5py.File(file_name, 'w') as h5file:
            # dataSet
            h5file.create_dataset('/dataSet', data=self.dataSet)

            # metaData
            md_keys = [m['key'] for m in self.metaData]
            md_desc = [m['description'] for m in self.metaData]
            md_vals = np.array([m['value'] for m in self.metaData], dtype=np.float)

            h5file.create_group('/metaData')
            h5file.create_dataset('/metaData/key', data=md_keys)
            h5file.create_dataset('/metaData/description', data=md_desc)
            h5file.create_dataset('/metaData/value', data=md_vals)


    def __load_npy(self, load_filename):
        """
        Load dataSet and metaData from Npy file
        """

        dat = np.load(load_filename)
        dicdat = dat.item()

        self.dataSet = dicdat["dataSet"]
        self.metaData = dicdat["metaData"]


    def __load_mat(self, load_filename):
        """
        Load dataSet and metaData from Matlab file
        """

        dat = sio.loadmat(load_filename)

        md_keys = [str(i[0]).strip() for i in np.asarray(dat["metaData"]['key'][0, 0])[0].tolist()]
        md_descs = [str(i[0]).strip() for i in np.asarray(dat["metaData"]['description'][0, 0])[0].tolist()]
        md_values = np.asarray(dat["metaData"]['value'][0, 0])

        self.dataSet = np.asarray(dat["dataSet"])

        for k, v, d in zip(md_keys, md_values, md_descs):
            self.add_metadata(k, v, d)


    def __load_h5(self, load_filename):
        """
        Load dataSet and metaData from HDF5 file
        """

        dat = h5py.File(load_filename)

        md_keys = dat["metaData"]['key'][:].tolist()
        md_descs = dat["metaData"]['description'][:].tolist()
        md_values = dat["metaData"]['value']

        self.dataSet = np.asarray(dat["dataSet"])

        for k, v, d in zip(md_keys, md_values, md_descs):
            self.add_metadata(k, v, d)


    def __get_filetype(self, file_name):
        """
        Return the type of `file_name` based on the file extension
        """
        _, ext = os.path.splitext(file_name)

        if ext == ".npy":
            file_type = "Npy"
        elif ext == ".mat":
            file_type = "Matlab"
        elif ext == ".h5":
            file_type = "HDF5"
        else:
            raise ValueError("Unknown file extension: %s" % (ext))

        return file_type
