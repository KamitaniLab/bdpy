from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.stimulus_domain import Domain, image_domain


class EncoderBase(nn.Module):
    """Encoder network module.

    Parameters
    ----------
    feature_network : nn.Module
        Feature network. This network should have a method `forward` that takes
        an image tensor and propagates it through the network.
    layer_names : list[str]
        Layer names to extract features from.
    domain : Domain
        Domain of the input images to receive.
    device : torch.device
        Device to use.
    """

    def __init__(
        self,
        feature_network: nn.Module,
        layer_names: Iterable[str],
        domain: Domain,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        self._feature_extractor = FeatureExtractor(
            encoder=feature_network, layers=layer_names, detach=False, device=device
        )
        self._domain = domain
        self._feature_network = self._feature_extractor._encoder

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the encoder network.

        Parameters
        ----------
        images : torch.Tensor
            Images.

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
) -> EncoderBase:
    """Build an encoder network.

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
    EncoderBase
        Encoder network.
    """
    return EncoderBase(feature_network, layer_names, domain, device)
