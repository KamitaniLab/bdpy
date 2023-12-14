from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    import torch


class Domain(nn.Module, ABC):
    """Base class for stimulus domain.

    This class is used to convert stimulus between each domain and library's internal common space.
    """

    @abstractmethod
    def send(self, x: torch.Tensor) -> torch.Tensor:
        """Send stimulus to the internal common space from each domain.

        Parameters
        ----------
        x : torch.Tensor
            Stimulus in the original domain.

        Returns
        -------
        torch.Tensor
            Stimulus in the internal common space.
        """
        pass

    @abstractmethod
    def receive(self, x: torch.Tensor) -> torch.Tensor:
        """Receive stimulus from the internal common space to each domain.

        Parameters
        ----------
        x : torch.Tensor
            Stimulus in the internal common space.

        Returns
        -------
        torch.Tensor
            Stimulus in the original domain.
        """
        pass


class IrreversibleDomain(Domain):
    """The domain which cannot be reversed.

    This class is used to convert stimulus between each domain and library's
    internal common space. Note that the subclasses of this class do not
    guarantee the reversibility of `send` and `receive` methods.
    """

    def send(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def receive(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ComposedDomain(Domain):
    """The domain composed of multiple domains."""

    def __init__(self, domains: Iterable[Domain]) -> None:
        super().__init__()
        self.domains = nn.ModuleList(domains)

    def send(self, x: torch.Tensor) -> torch.Tensor:
        for domain in reversed(self.domains):
            x = domain.send(x)
        return x

    def receive(self, x: torch.Tensor) -> torch.Tensor:
        for domain in self.domains:
            x = domain.receive(x)
        return x
