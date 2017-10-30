'''DataStore class

This file is a part of BdPy.
'''


from __future__ import print_function

import os
import glob
import re

import numpy as np
import scipy.io as sio
import h5py


__all__ = ['DataStore']


class DataStore(object):
    '''Data store class.

    Parameters
    ----------
    dpath : str or list
        Path(s) to data directory(ies).
    file_type : {'mat', 'mat_hdf5'}
        Data file format.
    pattern : str, regular expression pattern
        Regular expression pattern to parse file names (paths).
    extractor : func, optional
        Function to extract data from files.

    Example
    -------

    # Suppose you have mat files in `/path/to/data/dir/`. Each file has name
    # such as `subject1_V1.mat`, and contains variable `data`.

    datastore = DataStore('/path/to/data/dir/',
                          file_type='mat',
                          pattern='.*/(.*?)_(.*?).mat',
                          extractor=lambda x: x['data'])

    dat = datastore.get('subject1', 'V1')

    TODO
    ----
    - Add input checks.
    - Add recursive file search.
    - Add default file name pattern (`pattern`).
    '''

    def __init__(self,
                 dpath=None, file_type=None,
                 pattern=None, extractor=None):
        self.__key_sep = '/'

        if isinstance(dpath, str):
            dpath = [dpath]

        self.root_path = dpath
        self.file_type = file_type
        self.pattern = pattern
        self.extractor = extractor

        self.n_keys = self.__get_key_num(self.pattern)

        self.file_dict = {}

        if self.root_path is not None:
            for p in self.root_path:
                self.__parse_datafiles(p)

    def get(self, *keys):
        '''Get data specified by keys.

        Parameters
        ----------
        keys : str
            Keys to specify data.

        Returns
        -------
        All variables in a file (dict) or extracted data.
        '''

        fpath = self.__get_file_from_keys(keys)
        dat = self.__load_data(fpath, self.extractor)

        return dat

    def __get_file_from_keys(self, keys_lst):
        '''Get file path specified by `keys_lst`.'''
        key = self.__key_sep.join(keys_lst)
        return self.file_dict[key]

    def __load_data(self, fpath, extractor):
        '''Load data in `fpath`.'''
        print('Loading ' + fpath)

        if self.file_type is None:
            raise RuntimeError('File type unspecified')
        if self.file_type == 'mat':
            dat = self.__load_data_mat(fpath, extractor)
        elif self.file_type == 'mat_hdf5':
            dat = self.__load_data_mat_hdf5(fpath, extractor)
        else:
            raise ValueError('Unknown file type: %s' % self.file_type)

        return dat

    def __load_data_mat(self, fpath, extractor):
        if extractor is None:
            return sio.loadmat(fpath)
        else:
            return extractor(sio.loadmat(fpath))

    def __load_data_mat_hdf5(self, fpath, extractor):
        if extractor is None:
            with h5py.File(fpath, 'r') as f:
                return f
        else:
            with h5py.File(fpath, 'r') as f:
                return extractor(f)

    def __get_key_num(self, pat):
        '''Return no. of keys.'''
        n_keys = len(re.findall('\(.*?\)', pat))
        return n_keys

    def __parse_datafiles(self, dpath):
        '''Parse files in `dpath`.'''

        print('Searching %s' % dpath)

        if not os.path.isdir(dpath):
            raise ValueError('Invalid directory path: %s' % dpath)

        if self.file_type is None:
            ext = ''
        elif self.file_type == 'mat':
            ext = '.mat'
        elif self.file_type == 'mat_hdf5':
            ext = '.mat'
        else:
            raise ValueError('Unknown file type: %s' % self.file_type)

        parse = re.compile(self.pattern)

        for f in glob.glob(dpath + '/*' + ext):
            m = parse.match(f)
            if m:
                key_list = [m.group(i) for i in range(1, self.n_keys + 1)]
                key = self.__key_sep.join(key_list)
                self.file_dict.update({key: f})
