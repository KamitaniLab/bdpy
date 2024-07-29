from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
import torch.nn as nn
from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.domain import Domain, InternalDomain


class BaseEncoder(ABC):
    """Encoder network module."""

    @abstractmethod
    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode images as a hierarchical feature representation.

        Parameters
        ----------
        images : torch.Tensor
            Images.

        Returns
        -------
        dict[str, torch.Tensor]
            Features indexed by the layer names.
        """
        pass

    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Call self.encode.

        Parameters
        ----------
        images : torch.Tensor
            Images on the libraries internal domain.

        Returns
        -------
        dict[str, torch.Tensor]
            Features indexed by the layer names.
        """
        return self.encode(images)


class NNModuleEncoder(BaseEncoder, nn.Module):
    """Encoder network module subclassed from nn.Module."""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Call self.encode.

        Parameters
        ----------
        images : torch.Tensor
            Images on the library's internal domain.

        Returns
        -------
        dict[str, torch.Tensor]
            Features indexed by the layer names.
        """
        return self.encode(images)

    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return nn.Module.__call__(self, images)


class SimpleEncoder(NNModuleEncoder):
    """Encoder network module with a naive feature extractor.

    Parameters
    ----------
    feature_network : nn.Module
        Feature network. This network should have a method `forward` that takes
        an image tensor and propagates it through the network.
    layer_names : list[str]
        Layer names to extract features from.
    domain : Domain, optional
        Domain of the input stimuli to receive. (default: InternalDomain())

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.modules.encoder import SimpleEncoder
    >>> feature_network = nn.Sequential(
    ...     nn.Conv2d(3, 3, 3),
    ...     nn.ReLU(),
    ... )
    >>> encoder = SimpleEncoder(feature_network, ['[0]'])
    >>> image = torch.randn(1, 3, 64, 64)
    >>> features = encoder(image)
    >>> features['[0]'].shape
    torch.Size([1, 3, 62, 62])
    """

    def __init__(
        self,
        feature_network: nn.Module,
        layer_names: Iterable[str],
        domain: Domain = InternalDomain(),
    ) -> None:
        super().__init__()
        self._feature_extractor = FeatureExtractor(
            encoder=feature_network, layers=layer_names, detach=False, device=None
        )
        self._domain = domain
        self._feature_network = self._feature_extractor._encoder

    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode images as a hierarchical feature representation.

        Parameters
        ----------
        images : torch.Tensor
            Images on the libraries internal domain.

        Returns
        -------
        dict[str, torch.Tensor]
            Features indexed by the layer names.
        """
        images = self._domain.receive(images)
        return self._feature_extractor(images)


def build_encoder(
    feature_network: nn.Module,
    layer_names: Iterable[str],
    domain: Domain = InternalDomain(),
) -> BaseEncoder:
    """Build an encoder network with a naive feature extractor.

    The function builds an encoder module from a feature network that takes
    images on its own domain as input and processes them. The encoder module
    receives images on the library's internal domain and returns features on the
    library's internal domain indexed by `layer_names`. `domain` is used to
    convert the input images to the feature network's domain from the library's
    internal domain.

    Parameters
    ----------
    feature_network : nn.Module
        Feature network. This network should have a method `forward` that takes
        an image tensor and propagates it through the network. The images should
        be on the network's own domain.
    layer_names : list[str]
        Layer names to extract features from.
    domain : Domain, optional
        Domain of the input stimuli to receive (default: InternalDomain()).
        One needs to specify the domain that corresponds to the feature network's
        input domain.

    Returns
    -------
    BaseEncoder
        Encoder network.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.modules.encoder import build_encoder
    >>> feature_network = nn.Sequential(
    ...     nn.Conv2d(3, 3, 3),
    ...     nn.ReLU(),
    ... )
    >>> encoder = build_encoder(feature_network, layer_names=['[0]'])
    >>> image = torch.randn(1, 3, 64, 64)
    >>> features = encoder(image)
    >>> features['[0]'].shape
    torch.Size([1, 3, 62, 62])
    """
    return SimpleEncoder(feature_network, layer_names, domain)
