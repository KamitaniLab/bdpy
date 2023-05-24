'''
Base classes.
'''


__all__ = [
    'DnnFeatureExtractorBase',
]


from typing import Any, Type, Iterable, List, Dict, Tuple, Callable, Union, Optional

import os

import numpy as np
import torch
import torch.nn as nn


_tensor_t = Union[np.ndarray, torch.Tensor]


class DnnFeatureExtractorBase(object):
    '''
    Base class for PyTorch DNN feature extractors.

    '''

    def __init__(self, model: nn.Module = None, model_cls: Type[nn.Module] = None, layers: Iterable[str] = [], device: str = 'cpu', init_args={}):
        self.model = model
        self.model_cls = model_cls
        self.layers = layers
        self.device = torch.device(device)

        self.init(**init_args)

        self.model.to(self.device)

    def init(self, **kwargs) -> None:
        '''
        Custom initialization method.
        `init_args` in `__init__()` is passed to this function.
        '''
        return None

    def preprocess(self, x: Any, **kwargs) -> Any:
        '''
        Preprocesses the input for the DNN model.
        '''
        return x

    def extract_features(self, x: Any, **kwargs) -> Dict[str, _tensor_t]:
        '''
        Extracts features from the given input using the DNN model.
        '''
        raise NotImplementedError("Subclass must implement extract_features method.")

    def __call__(self, x: Any, **kwargs) -> Dict[str, _tensor_t]:
        return self.extract_features(self.preprocess(x), **kwargs)


