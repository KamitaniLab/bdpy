'''DNN features class


This file is a part of BdPy.
'''


from __future__ import print_function


__all__ = ['Features']


import os
import glob

import numpy as np
import scipy.io as sio


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

    def __init__(self, dpath=[], ext='mat'):
        if type(dpath) != list:
            dpath = [dpath]
        self.__dpath = dpath

        self.__feature_file_table = {} # Stimulus feature file tables
        self.__labels = []             # Stimulus labels
        self.__index = []              # Stimulus index (one-based)
        self.__layers = []             # DNN layers
        self.__collect_feature_files(ext=ext)

        self.__c_feature_name = None  # Loaded layer
        self.__features = None        # Loaded features

    @property
    def labels(self):
        return self.__labels

    @property
    def index(self):
        return self.__index

    @property
    def layers(self):
        return self.__layers

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

        self.__features = np.vstack(
            [sio.loadmat(self.__feature_file_table[layer][label])['feat']
             for label in self.__labels]
        )
        self.__c_feature_name = layer

        return self.__features 

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
            files = glob.glob(os.path.join(lay_dir, '*.' + ext))
            labels_t = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])
            if not labels:
                labels = labels_t
            else:
                if labels != labels_t:
                    raise RuntimeError('Invalid feature file in %s ' % dpath)
        return labels
