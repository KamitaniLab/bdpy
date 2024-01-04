from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TypeVar, Generic
import warnings

import torch.nn as nn

_T = TypeVar("_T")


class Domain(nn.Module, ABC, Generic[_T]):
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
    def send(self, x: _T) -> _T:
        """Send stimulus to the internal common space from each domain.

        Parameters
        ----------
        x : _T
            Data in the original domain.

        Returns
        -------
        _T
            Data in the internal common space.
        """
        pass

    @abstractmethod
    def receive(self, x: _T) -> _T:
        """Receive data from the internal common space to each domain.

        Parameters
        ----------
        x : _T
            Data in the internal common space.

        Returns
        -------
        _T
            Data in the original domain.
        """
        pass


class InternalDomain(Domain, Generic[_T]):
    """The internal common space.

    The domain class which defines the internal common space. This class
    receives and sends data as it is.
    """

    def send(self, x: _T) -> _T:
        return x

    def receive(self, x: _T) -> _T:
        return x


class IrreversibleDomain(Domain, Generic[_T]):
    """The domain which cannot be reversed.

    This class is used to convert data between each domain and library's
    internal common space. Note that the subclasses of this class do not
    guarantee the reversibility of `send` and `receive` methods.
    """

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            f"{self.__class__.__name__} is an irreversible domain. " \
            "It does not guarantee the reversibility of `send` and `receive` " \
            "methods. Please use the combination of `send` and `receive` methods " \
            "with caution.",
            RuntimeWarning,
        )

    def send(self, x: _T) -> _T:
        return x

    def receive(self, x: _T) -> _T:
        return x


class ComposedDomain(Domain, Generic[_T]):
    """The domain composed of multiple sub-domains.

    Suppose we have list of domain objects `domains = [d_0, d_1, ..., d_n]`.
    Then, `ComposedDomain(domains)` accesses the data in the original domain `D`
    as `d_n.receive . ... d_1.receive . d_0.receive(x)` from the internal common space `D_0`.

    Parameters
    ----------
    domains : Iterable[Domain]
        Sub-domains to compose.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from bdpy.dl.torch.domain import ComposedDomain
    >>> from bdpy.dl.torch.domain.image_domain import AffineDomain, BGRDomain
    >>> composed_domain = ComposedDomain([
    ...     AffineDomain(0.5, 1),
    ...     BGRDomain(),
    ... ])
    >>> image = torch.randn(1, 3, 64, 64).clamp(-0.5, 0.5)
    >>> image.shape
    torch.Size([1, 3, 64, 64])
    >>> composed_domain.send(image).shape
    torch.Size([1, 3, 64, 64])
    >>> print(composed_domain.send(image).min().item(), composed_domain.send(image).max().item())
    0.0 1.0
    """

    def __init__(self, domains: Iterable[Domain]) -> None:
        super().__init__()
        self.domains = nn.ModuleList(domains)

    def send(self, x: _T) -> _T:
        for domain in reversed(self.domains):
            x = domain.send(x)
        return x

    def receive(self, x: _T) -> _T:
        for domain in self.domains:
            x = domain.receive(x)
        return x


class KeyValueDomain(Domain, Generic[_T]):
    """The domain which converts key-value pairs.

    This class is used to convert key-value pairs between each domain and library's
    internal common space.

    Parameters
    ----------
    domain_mapper : dict[str, Domain]
        Dictionary that maps keys to domains.
    """

    def __init__(self, domain_mapper: dict[str, Domain]) -> None:
        super().__init__()
        self.domain_mapper = domain_mapper

    def send(self, x: dict[str, _T]) -> dict[str, _T]:
        return {
            key: self.domain_mapper[key].send(value) for key, value in x.items()
        }

    def receive(self, x: dict[str, _T]) -> dict[str, _T]:
        return {
            key: self.domain_mapper[key].receive(value) for key, value in x.items()
        }
