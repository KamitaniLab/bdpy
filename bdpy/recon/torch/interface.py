from typing import Dict, Protocol, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    FeatureType = Dict[str, torch.Tensor]


class Encoder(Protocol):
    def __call__(self, image: torch.Tensor) -> FeatureType:
        ...


class Generator(Protocol):
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        ...

    def parameters(self, recurse: bool = True) -> Iterable[torch.Tensor]:
        ...

    def reset_state(self) -> None:
        ...


class Latent(Protocol):
    def __call__(self) -> torch.Tensor:
        ...

    def parameters(self, recurse: bool = True) -> Iterable[torch.Tensor]:
        ...

    def reset_state(self) -> None:
        ...


class Critic(Protocol):
    def __call__(self, features: FeatureType, target_features: FeatureType) -> torch.Tensor:
        ...
