'''DNN features class


This file is a part of BdPy.
'''


from __future__ import print_function


__all__ = ['Features']


import os
import glob

import numpy as np
import scipy.io as sio
import hdf5storage


class Features(object):
    '''DNN features class.

    Parameters
    ----------
    dpath: str or list
       (List of) DNN feature directory(ies)
    ext: str
        DNN feature file extension (default: mat)

    Attributes
    ----------
    labels: list
       List of stimulus labels
    index: list
       List of stimulus index (one-based)
    layers: list
       List of DNN layers
    '''

    def __init__(self, dpath=[], ext='mat', feature_index=None):
        if type(dpath) != list:
            dpath = [dpath]
        self.__dpath = dpath
        self.__feat_index_table = feature_index

        self.__feature_file_table = {} # Stimulus feature file tables
        self.__labels = []             # Stimulus labels
        self.__index = []              # Stimulus index (one-based)
        self.__feature_index = []      # Feature (unit) index
        self.__layers = []             # DNN layers
        self.__collect_feature_files(ext=ext)

        self.__c_feature_name = None  # Loaded layer
        self.__features = None        # Loaded features
        self.__feature_index = None   # Indexes of loaded features

        if self.__feat_index_table is not None:
            if not os.path.exists(self.__feat_index_table):
                raise RuntimeError('%s do not exist' % self.__feat_index_table)
            self.__feat_index_table = hdf5storage.loadmat(self.__feat_index_table)['index']

    @property
    def labels(self):
        return self.__labels

    @property
    def index(self):
        return self.__index

    @property
    def layers(self):
        return self.__layers

    @property
    def feature_index(self):
        return self.__feature_index

    def get_features(self, layer):
        '''Return features in `layer`.

        Parameters
        ----------
        layer: str
            DNN layer

        Returns
        -------
        numpy.ndarray, shape=(n_samples, shape_layers)
            DNN features
        '''

        if layer == self.__c_feature_name:
            return self.__features

        try:
            self.__features = np.vstack(
                [sio.loadmat(self.__feature_file_table[layer][label])['feat']
                 for label in self.__labels]
            )
        except NotImplementedError:
            self.__features = np.vstack(
                [hdf5storage.loadmat(self.__feature_file_table[layer][label])['feat']
                 for label in self.__labels]
            )

        self.__c_feature_name = layer

        if self.__feat_index_table is not None:
            # Select features by index
            self.__feature_index = self.__feat_index_table[layer]
            n_sample = self.__features.shape[0]
            n_feat = np.array(self.__features.shape[1:]).prod()

            self.__features = self.__features.reshape([n_sample, n_feat], order='C')[:, self.__feature_index]

        return self.__features

    def save_feature_index(self, fname):
        '''Save feature indexes in `fname`'''
        if len(self.__feature_index) == 0:
            raise RuntimeError('No feature index specified')

        hdf5storage.savemat(fname,
                            {'index': self.__feature_index},
                            format='7.3', oned_as='column',
                            store_python_metadata=True)

    def __collect_feature_files(self, ext='mat'):
        dpath_lst = self.__dpath

        # List-up layers and stimulus labels
        label_dir = {}
        for dpath in dpath_lst:
            # List-up layers
            self.__layers = self.__get_layers(dpath)

            # List-up stimulus labels
            labels_in_dir = self.__get_labels(dpath, self.__layers, ext=ext)
            label_dir.update({label: dpath for label in labels_in_dir})
            self.__labels += labels_in_dir

        self.__index = np.arange(len(self.__labels)) + 1

        # List-up feature files
        for lay in self.__layers:
            self.__feature_file_table.update(
                {
                    lay:
                    {
                        label:
                         os.path.join(label_dir[label], lay, label + '.' + ext)
                         for label in self.__labels
                    }
                })

        return None

    def __get_layers(self, dpath):
        layers = sorted([d for d in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, d))])
        if self.__layers and (layers != self.__layers):
            raise RuntimeError('Invalid layers in %s' % dpath)
        return layers

    def __get_labels(self, dpath, layers, ext='mat'):
        labels = []
        for lay in layers:
            lay_dir = os.path.join(dpath, lay)
            lay_dir = lay_dir.replace('[', '[[]') # Use glob.escape for Python 3.4 or later
            files = glob.glob(os.path.join(lay_dir, '*.' + ext))
            labels_t = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])
            if not labels:
                labels = labels_t
            else:
                if labels != labels_t:
                    raise RuntimeError('Invalid feature file in %s ' % dpath)
        return labels
