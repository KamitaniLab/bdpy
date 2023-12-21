from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator

import torch
import torch.nn as nn


class BaseLatent(ABC):
    """Latent variable module."""

    @abstractmethod
    def reset_states(self) -> None:
        """Reset the state of the latent variable."""
        pass

    @abstractmethod
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return the parameters of the latent variable."""
        pass

    @abstractmethod
    def generate(self) -> torch.Tensor:
        """Generate a latent variable.

        Returns
        -------
        torch.Tensor
            Latent variable.
        """
        pass

    def __call__(self) -> torch.Tensor:
        """Call self.generate.

        Returns
        -------
        torch.Tensor
            Latent variable.
        """
        return self.generate()


class NNModuleLatent(BaseLatent, nn.Module):
    """Latent variable module uses __call__ method and parameters method of nn.Module."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return nn.Module.parameters(self, recurse=recurse)

    def __call__(self) -> torch.Tensor:
        return nn.Module.__call__(self)

    def forward(self) -> torch.Tensor:
        return self.generate()


class ArbitraryLatent(NNModuleLatent):
    """Latent variable with arbitrary shape and initialization function.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the latent variable including the batch dimension.
    init_fn : Callable[[torch.Tensor], None]
        Function to initialize the latent variable.

    Examples
    --------
    >>> from functools import partial
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.modules.latent import ArbitraryLatent
    >>> latent = ArbitraryLatent((1, 3, 64, 64), partial(nn.init.normal_, mean=0, std=1))
    >>> latent().shape
    torch.Size([1, 3, 64, 64])
    """

    def __init__(self, shape: tuple[int, ...], init_fn: Callable[[torch.Tensor], None]) -> None:
        super().__init__()
        self._shape = shape
        self._init_fn = init_fn
        self._latent = nn.Parameter(torch.empty(shape))

    def reset_states(self) -> None:
        """Reset the state of the latent variable."""
        self._init_fn(self._latent)

    def generate(self) -> torch.Tensor:
        """Generate a latent variable.

        Returns
        -------
        torch.Tensor
            Latent variable.
        """
        return self._latent
