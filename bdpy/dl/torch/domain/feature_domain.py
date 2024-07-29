from __future__ import annotations

from typing import Dict

import torch

from .core import Domain

_FeatureType = Dict[str, torch.Tensor]


def _lnd2nld(feature: torch.Tensor) -> torch.Tensor:
    """Convert features having the shape of (L, N, D) to (N, L, D)."""
    return feature.permute(1, 0, 2)

def _nld2lnd(feature: torch.Tensor) -> torch.Tensor:
    """Convert features having the shape of (N, L, D) to (L, N, D)."""
    return feature.permute(1, 0, 2)


class ArbitraryFeatureKeyDomain(Domain):
    def __init__(
        self,
        to_internal: dict[str, str] | None = None,
        to_self: dict[str, str] | None = None,
    ):
        super().__init__()

        if to_internal is None and to_self is None:
            raise ValueError("Either to_internal or to_self must be specified.")

        if to_internal is None:
            to_internal = {value: key for key, value in to_self.items()}
        elif to_self is None:
            to_self = {value: key for key, value in to_internal.items()}

        self._to_internal = to_internal
        self._to_self = to_self

    def send(self, features: _FeatureType) -> _FeatureType:
        return {self._to_internal.get(key, key): value for key, value in features.items()}

    def receive(self, features: _FeatureType) -> _FeatureType:
        return {self._to_self.get(key, key): value for key, value in features.items()}
