from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    import torch


class Domain(nn.Module, ABC):
    """Base class for stimulus domain.

    This class is used to convert data between each domain and library's internal common space.
    Suppose that we have two functions `f: X -> Y_1` and `g: Y_2 -> Z` and want to compose them.
    Here, `X`, `Y_1`, `Y_2`, and `Z` are different domains and assume that `Y_1` and `Y_2` are
    the similar domain that can be converted to each other.
    Then, we can compose `f` and `g` as `g . t . f(x)`, where `t: Y_1 -> Y_2` is the domain
    conversion function. This class is used to implement `t`.

    The subclasses of this class should implement `send` and `receive` methods. The `send` method
    converts data from the original domain (`Y_1` or `Y_2`) to the internal common space (`Y_0`),
    and the `receive` method converts data from the internal common space to the original domain.
    By implementing domain class for `Y_1` and `Y_2`, we can construct the domain conversion function
    `t` as `t = Y_2.receive . Y_1.send`.

    Note that the subclasses of this class do not necessarily guarantee the reversibility of `send`
    and `receive` methods. If the domain conversion is irreversible, the subclasses should inherit
    `IrreversibleDomain` class instead of this class.
    """

    @abstractmethod
    def send(self, x: torch.Tensor) -> torch.Tensor:
        """Send stimulus to the internal common space from each domain.

        Parameters
        ----------
        x : torch.Tensor
            Data in the original domain.

        Returns
        -------
        torch.Tensor
            Data in the internal common space.
        """
        pass

    @abstractmethod
    def receive(self, x: torch.Tensor) -> torch.Tensor:
        """Receive data from the internal common space to each domain.

        Parameters
        ----------
        x : torch.Tensor
            Data in the internal common space.

        Returns
        -------
        torch.Tensor
            Data in the original domain.
        """
        pass


class IrreversibleDomain(Domain):
    """The domain which cannot be reversed.

    This class is used to convert data between each domain and library's
    internal common space. Note that the subclasses of this class do not
    guarantee the reversibility of `send` and `receive` methods.
    """

    def send(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def receive(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ComposedDomain(Domain):
    """The domain composed of multiple sub-domains."""

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
