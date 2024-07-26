from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable

import torch
import torch.nn as nn

from bdpy.task.callback import CallbackHandler, BaseCallback


_FeatureType = Dict[str, torch.Tensor]


class BaseCritic(ABC):
    """Critic network module."""

    def __init__(self, callbacks: BaseCallback | Iterable[BaseCallback] | None = None) -> None:
        self._callback_handler = CallbackHandler(callbacks)

    def __call__(self, features: _FeatureType, target_features: _FeatureType) -> torch.Tensor:
        """Call self.compare.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Features indexed by the layer names.
        target_features : dict[str, torch.Tensor]
            Target features indexed by the layer names.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return self.evaluate(features, target_features)

    @abstractmethod
    def evaluate(
        self,
        features: _FeatureType,
        target_features: _FeatureType,
    ) -> torch.Tensor:
        """Compute the total loss value given the features and the target features.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Features indexed by the layer names.
        target_features : dict[str, torch.Tensor]
            Target features indexed by the layer names.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        pass


class NNModuleCritic(BaseCritic, nn.Module):
    """Critic network module uses __call__ method of nn.Module."""
    def __init__(self, callbacks: BaseCallback | Iterable[BaseCallback] | None = None) -> None:
        BaseCritic.__init__(self, callbacks)
        nn.Module.__init__(self)

    def __call__(self, features: _FeatureType, target_features: _FeatureType) -> torch.Tensor:
        return nn.Module.__call__(self, features, target_features)

    def forward(self, features: _FeatureType, target_features: _FeatureType) -> torch.Tensor:
        return self.evaluate(features, target_features)


class LayerWiseAverageCritic(NNModuleCritic):
    """Compute the average of the layer-wise loss values."""

    def evaluate(
        self,
        features: _FeatureType,
        target_features: _FeatureType,
    ) -> torch.Tensor:
        """Compute the total loss value given the features and the target features.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Features indexed by the layer names.
        target_features : dict[str, torch.Tensor]
            Target features indexed by the layer names.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        loss = 0.0
        counts = 0
        for layer_name, feature in features.items():
            target_feature = target_features[layer_name]
            layer_wise_loss = self.compare_layer(
                feature, target_feature, layer_name=layer_name
            )
            self._callback_handler.fire(
                "on_layerwise_loss_calculated",
                layer_name=layer_name,
                layer_loss=layer_wise_loss,
            )
            loss += layer_wise_loss
            counts += 1
        return loss / counts

    @abstractmethod
    def compare_layer(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """Loss function per layer.

        Parameters
        ----------
        feature : torch.Tensor
            Feature tensor of the layer specified by `layer_name`.
        target_feature : torch.Tensor
            Target feature tensor of the layer specified by `layer_name`.
        layer_name : str
            Layer name.

        Returns
        -------
        torch.Tensor
            Loss value of the layer specified by `layer_name`.
        """
        pass


class MSE(LayerWiseAverageCritic):
    """MSE loss."""

    def compare_layer(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """Loss function per layer.

        Parameters
        ----------
        feature : torch.Tensor
            Feature tensor of the layer specified by `layer_name`.
        target_feature : torch.Tensor
            Target feature tensor of the layer specified by `layer_name`.
        layer_name : str
            Layer name.

        Returns
        -------
        torch.Tensor
            Loss value of the layer specified by `layer_name`.
        """
        return ((feature - target_feature) ** 2).sum(
            dim=tuple(range(1, feature.ndim))
        )


class TargetNormalizedMSE(LayerWiseAverageCritic):
    """MSE loss divided by the squared norm of the target feature."""

    def compare_layer(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """Loss function per layer.

        Parameters
        ----------
        feature : torch.Tensor
            Feature tensor of the layer specified by `layer_name`.
        target_feature : torch.Tensor
            Target feature tensor of the layer specified by `layer_name`.
        layer_name : str
            Layer name.

        Returns
        -------
        torch.Tensor
            Loss value of the layer specified by `layer_name`.
        """
        squared_norm = (target_feature ** 2).sum(dim=tuple(range(1, target_feature.ndim)))
        return ((feature - target_feature) ** 2).sum(
            dim=tuple(range(1, feature.ndim))
        ) / squared_norm
