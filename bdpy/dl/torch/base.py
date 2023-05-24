'''
Base classes.
'''


__all__ = [
    'DnnFeatureExtractorBase',
    'ReconstructionBase',
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

    def __init__(self, model: Optional[nn.Module] = None, model_cls: Optional[Type[nn.Module]] = None, layers: Iterable[str] = [], device: str = 'cpu', init_args={}):
        self.model = model
        self.model_cls = model_cls
        self.layers = layers
        self.device = torch.device(device)

        self.init(**init_args)

        if self.model is None:
            raise RuntimeError('`self.model` is None. You should define it it `init()`.')

        self.model.to(self.device)

    def init(self) -> None:
        '''
        Custom initialization method.
        `init_args` in `__init__()` is passed to this function.
        '''
        return None

    def preprocess(self, x: Any) -> Any:
        '''
        Preprocesses the input for the DNN model.
        '''
        return x

    def extract_features(self, x: Any) -> Dict[str, np.ndarray]:
        '''
        Extracts features from the given input using the DNN model.
        '''
        raise NotImplementedError("Subclass must implement extract_features method.")

    def __call__(self, x: Any, **kwargs) -> Dict[str, _tensor_t]:
        return self.extract_features(self.preprocess(x), **kwargs)


class ReconstructionBase(object):
    '''
    Base class for reconstruction.

    '''

    def __init__(self, model: Optional[nn.Module] = None, model_cls: Optional[Type[nn.Module]] = None, layers: Iterable[str] = [], device: str = 'cpu', init_args={}):
        self.model = model
        self.model_cls = model_cls
        self.layers = layers
        self.device = torch.device(device)

        self.init(**init_args)

        if self.model is None:
            raise RuntimeError('`self.model` is None. You should define it it `init()`.')

        self.model.to(self.device)

    def init(self) -> None:
        '''
        Custom initialization method.
        `init_args` in `__init__()` is passed to this function.
        '''
        return None

    def preprocess(self, x: Any) -> Any:
        '''
        Preprocesses the input for the DNN model.
        '''
        return x

    def reconstruct(self, x: Any) -> Any:
        '''
        Reconstruction from the given input.
        '''
        raise NotImplementedError("Subclass must implement reconstruct method.")

    def __call__(self, x: Any, **kwargs) -> Any:
        return self.reconstruct(self.preprocess(x), **kwargs)
