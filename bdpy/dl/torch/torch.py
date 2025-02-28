'''PyTorch module.'''

from __future__ import annotations

from typing import Iterable, List, Dict, Union, Tuple, Any, Callable, Optional
from collections import OrderedDict
import os
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from . import models

_tensor_t = Union[np.ndarray, torch.Tensor]


class _Hook:
    """Forward hook class for FeatureExtractor."""
    def __init__(self, layer_name, features_dict, detach):
        self.layer_name = layer_name
        self.features_dict = features_dict
        self.detach = detach

    def __call__(self, module, input, output):
        self.features_dict[self.layer_name] = output.detach() if self.detach else output


class FeatureExtractor:
    def __init__(
            self, 
            encoder: nn.Module, 
            layers: Iterable[str], 
            layer_mapping: Optional[Dict[str, str]] = None, 
            device: str = 'cpu',
            detach: bool = False,
            transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None, 
        ):
        """
        Initializes the feature extractor.
        
        Parameters
        ----------
        encoder : nn.Module 
            The encoder to extract features from.
        layers : Iterable[str]
            List of names of layers to extract features.
        layer_mapping : dict, optional
            Mapping of human-readable layer names to actual layer names.
        device : str, optional
            Device to run the encoder on. Defaults to 'cpu'.
        detach : bool, optional
            Whether to detach the extracted feature tensors. Defaults to False.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional 
            A function that receives extracted features dict and returns transformed features.
            Example usage: turning a tuple into a tensor on transformer models.
        """
        self.encoder = encoder
        self.layers = layers
        self.layer_mapping = layer_mapping if layer_mapping else {layer: layer for layer in layers}
        self.device = device
        self.detach = detach
        self.transform = transform

        self._features = OrderedDict()  # dict to store extracted features
        self._hook_handler = {}         # layer name -> hook handler object for deleting them
        self._register_hooks()

        self.encoder.to(self.device)

    def _register_hooks(self):
        """Registers forward hooks to extract features from specified layers."""
        for name in self.layers:
            mapped_name = self.layer_mapping[name]
            layer_obj = models._parse_layer_name(self.encoder, mapped_name)
            hook_handle = layer_obj.register_forward_hook(
                _Hook(name, self._features, self.detach)
            )
            self._hook_handler[name] = hook_handle  # Store the actual hook handle
    
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        return self.run(x)
    
    def run(self, x: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Feeds input through the encoder and extracts features.
        
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input tensor to be processed by the encoder.
        
        Returns:
        ----------
        dict[str, Union[torch.Tensor, np.ndarray]]
            A dictionary containing features extracted from the specified layers, 
            optionally transformed.
        """
        if not isinstance(x, torch.Tensor):
            # TODO: From legacy code. Is this necessary?
            warnings.warn(
                "In future versions, it will no longer be possible to input ndarrays " \
                "directly into FeatureExtractor. Please convert the input to torch.Tensor.",
                DeprecationWarning,
                stacklevel=2
            )
            x = torch.tensor(x[np.newaxis], device=self.device)

        self._features.clear()  # Clear previous feature maps
        _ = self.encoder(x)  # Forward pass to capture features

        # Shallow copy the dict to avoid features changed after returning
        features = self._features.copy()
        if self.transform:
            features = self.transform(features)
        if self.detach:
            # TODO: From legacy code. "detach" but it's actually also "numpy".
            features = {layer: feat.cpu().numpy() for layer, feat in features.items()}
        return features

    def __del__(self):
        """Removes all registered hooks when the instance is deleted."""
        for name, hook in self._hook_handler.items():
            hook.remove()
        self._hook_handler.clear()


class ImageDataset(torch.utils.data.Dataset):
    '''Pytoch dataset for images.'''

    def __init__(
            self, images: List[str],
            labels: Optional[List[str]] = None,
            label_dirname: bool = False,
            resize: Optional[Tuple[int, int]] = None,
            shape: str = 'chw',
            transform: Optional[Callable[[_tensor_t], torch.Tensor]] = None,
            scale: float = 1,
            rgb_mean: Optional[List[float]] = None,
            preload: bool = False,
            preload_limit: float = np.inf
    ):
        '''
        Parameters
        ----------
        images : List[str]
            List of image file paths.
        labels : List[str], optional
            List of image labels (default: image file names).
        label_dirname : bool, optional
            Use directory names as labels if True (default: False).
        resize : None or tuple, optional
            If not None, images will be resized by the specified size.
        shape : str ({'chw', 'hwc', ...}), optional
            Specify array shape (channel, hieght, and width).
        transform : Callable[[Union[np.ndarray, torch.Tensor]], torch.Tensor], optional
            Transformers (applied after resizing, reshaping, ans scaling to [0, 1])
        scale : optional
            Image intensity is scaled to [0, scale] (default: 1).
        rgb_mean : list([r, g, b]), optional
            Image values are centered by the specified mean (after scaling) (default: None).
        preload : bool, optional
            Pre-load images (default: False).
        preload_limit : float
            Memory size limit of preloading in GiB (default: unlimited).

        Note
        ----
        - Images are converted to RGB. Alpha channels in RGBA images are ignored.
        '''

        warnings.warn(
            "dl.torch.torch.ImageDataset is deprecated. Please consider using " \
            "bdpy.dl.torch.dataset.ImageDataset instead.",
            DeprecationWarning,
            stacklevel=2
        )

        self.transform = transform
        # Custom transforms
        self.__shape = shape
        self.__resize = resize
        self.__scale = scale
        self.__rgb_mean = rgb_mean

        self.__data = {}
        preload_size = 0
        image_labels = []
        for i, imf in enumerate(images):
            # TODO: validate the image file
            if label_dirname:
                image_labels.append(os.path.basename(os.path.dirname(imf)))
            else:
                image_labels.append(os.path.basename(imf))
            if preload:
                data = self.__load_image(imf)
                data_size = data.size * data.itemsize
                if preload_size + data_size > preload_limit * (1024 ** 3):
                    preload = False
                    continue
                self.__data.update({i: data})
                preload_size += data_size

        self.data_path = images
        if not labels is None:
            self.labels = labels
        else:
            self.labels = image_labels
        self.n_sample = len(images)

    def __len__(self) -> int:
        return self.n_sample

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        if idx in self.__data:
            data = self.__data[idx]
        else:
            data = self.__load_image(self.data_path[idx])

        if self.transform is not None:
            data = self.transform(data)
        else:
            data = torch.Tensor(data)

        label = self.labels[idx]

        return data, label

    def __load_image(self, fpath: str) -> np.ndarray:
        img = Image.open(fpath)

        # CMYK, RGBA --> RGB
        if img.mode == 'CMYK':
            img = img.convert('RGB')
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        data = np.asarray(img)

        # Monotone to RGB
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=2)

        # Resize the image
        if not self.__resize is None:
            data = np.array(Image.fromarray(data).resize(self.__resize, resample=2))  # bicubic

        # Reshape
        s2d = {'h': 0, 'w': 1, 'c': 2}
        data = data.transpose((s2d[self.__shape[0]],
                               s2d[self.__shape[1]],
                               s2d[self.__shape[2]]))

        # Scaling to [0, scale]
        data = (data / 255.) * self.__scale

        # Centering
        if not self.__rgb_mean is None:
            data[0] -= self.__rgb_mean[0]
            data[1] -= self.__rgb_mean[1]
            data[2] -= self.__rgb_mean[2]

        return data
