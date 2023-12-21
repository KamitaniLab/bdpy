from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
import torch.nn as nn
from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.domain import Domain, image_domain


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


class SimpleEncoder(BaseEncoder):
    """Encoder network module with a naive feature extractor.

    Parameters
    ----------
    feature_network : nn.Module
        Feature network. This network should have a method `forward` that takes
        an image tensor and propagates it through the network.
    layer_names : list[str]
        Layer names to extract features from.
    domain : Domain, optional
        Domain of the input images to receive. (default: Zero2OneImageDomain())
    device : torch.device, optional
        Device to use. (default: "cpu").

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
        domain: Domain = image_domain.Zero2OneImageDomain(),
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self._feature_extractor = FeatureExtractor(
            encoder=feature_network, layers=layer_names, detach=False, device=device
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
    domain: Domain = image_domain.Zero2OneImageDomain(),
    device: str | torch.device = "cpu",
) -> BaseEncoder:
    """Build an encoder network with a naive feature extractor.

    Parameters
    ----------
    feature_network : nn.Module
        Feature network. This network should have a method `forward` that takes
        an image tensor and propagates it through the network.
    layer_names : list[str]
        Layer names to extract features from.
    domain : Domain, optional
        Domain of the input images to receive (default: Zero2OneImageDomain()).
    device : torch.device, optional
        Device to use. (default: "cpu").

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
    return SimpleEncoder(feature_network, layer_names, domain, device)
