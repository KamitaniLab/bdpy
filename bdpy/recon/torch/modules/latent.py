from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn


class BaseLatent(nn.Module, ABC):
    """Latent variable module."""

    @abstractmethod
    def reset_states(self) -> None:
        """Reset the state of the latent variable."""
        pass

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Generate a latent variable.

        Returns
        -------
        torch.Tensor
            Latent variable.
        """
        pass


class ArbitraryLatent(BaseLatent):
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
        self._latent = torch.empty(shape)

    def reset_states(self) -> None:
        """Reset the state of the latent variable."""
        self._init_fn(self._latent)

    def forward(self) -> torch.Tensor:
        """Generate a latent variable.

        Returns
        -------
        torch.Tensor
            Latent variable.
        """
        return self._latent
