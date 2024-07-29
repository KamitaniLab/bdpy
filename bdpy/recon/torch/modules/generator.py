from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator
import warnings

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from bdpy.dl.torch.domain import Domain, InternalDomain


def _get_reset_module_fn(module: nn.Module) -> Callable[[], None] | None:
    """Get the function to reset the parameters of the module."""
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        return reset_parameters
    # NOTE: This is needed for nn.MultiheadAttention
    reset_parameters = getattr(module, "_reset_parameters", None)
    if callable(reset_parameters):
        return reset_parameters
    return None


@torch.no_grad()
def call_reset_parameters(module: nn.Module) -> None:
    """Reset the parameters of the module."""
    warnings.warn(
        "`call_reset_parameters` calls the instance method named `reset_parameters` " \
        "or `_reset_parameters` of the module. This method does not guarantee that " \
        "all the parameters of the module are reset. Please use this method with " \
        "caution.",
        UserWarning,
        stacklevel=2,
    )
    reset_parameters = _get_reset_module_fn(module)
    if reset_parameters is not None:
        reset_parameters()


class BaseGenerator(ABC):
    """Generator module."""

    @abstractmethod
    def reset_states(self) -> None:
        """Reset the state of the generator."""
        pass

    @abstractmethod
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return the parameters of the generator."""
        pass

    @abstractmethod
    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate image given latent variable.

        Parameters
        ----------
        latent : torch.Tensor
            Latent variable.

        Returns
        -------
        torch.Tensor
            Generated image on the libraries internal domain.
        """
        pass

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Call self.generate.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Generated image. The generated images must be in the range [0, 1].
        """
        return self.generate(latent)


class NNModuleGenerator(BaseGenerator, nn.Module):
    """Generator module uses __call__ method and parameters method of nn.Module."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return nn.Module.parameters(self, recurse=recurse)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.generate(latent)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return nn.Module.__call__(self, latent)


class BareGenerator(NNModuleGenerator):
    """Bare generator module.

    This module does not have any trainable parameters.

    Parameters
    ----------
    activation : Callable[[torch.Tensor], torch.Tensor], optional
        Activation function to apply to the output of the generator, by default nn.Identity()

    Examples
    --------
    >>> import torch
    >>> from bdpy.recon.torch.modules.generator import BareGenerator
    >>> generator = BareGenerator(activation=torch.sigmoid)
    >>> latent = torch.randn(1, 3, 64, 64)
    >>> generated_image = generator(latent)
    >>> generated_image.shape
    torch.Size([1, 3, 64, 64])
    """

    def __init__(self, activation: Callable[[torch.Tensor], torch.Tensor] = nn.Identity()) -> None:
        """Initialize the generator."""
        super().__init__()
        self._activation = activation
        self._domain = InternalDomain()

    def reset_states(self) -> None:
        """Reset the state of the generator."""
        pass

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """Naively pass the latent vector to the activation function.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Generated image on the libraries internal domain.
        """
        return self._domain.send(self._activation(latent))


class DNNGenerator(NNModuleGenerator):
    """DNN generator module.

    This module has the generator network as a submodule and its parameters are
    trainable.

    Parameters
    ----------
    generator_network : nn.Module
        Generator network. This network should have a method `forward` that takes
        a latent vector and propagates it through the network.
    domain : Domain, optional
        Domain of the input stimuli to receive. (default: InternalDomain())
    reset_fn : Callable[[nn.Module], None], optional
        Function to reset the parameters of the generator network, by default
        call_reset_parameters.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.modules.generator import DNNGenerator
    >>> generator_network = nn.Sequential(
    ...     nn.ConvTranspose2d(3, 3, 3),
    ...     nn.ReLU(),
    ... )
    >>> generator = DNNGenerator(generator_network)
    >>> latent = torch.randn(1, 3, 64, 64)
    >>> generated_image = generator(latent)
    >>> generated_image.shape
    torch.Size([1, 3, 66, 66])
    """

    def __init__(
        self,
        generator_network: nn.Module,
        domain: Domain = InternalDomain(),
        reset_fn: Callable[[nn.Module], None] = call_reset_parameters,
    ) -> None:
        """Initialize the generator."""
        super().__init__()
        self._generator_network = generator_network
        self._domain = domain
        self._reset_fn = reset_fn

    def reset_states(self) -> None:
        """Reset the state of the generator."""
        self._generator_network.apply(self._reset_fn)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate image using the generator network.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Generated image on the libraries internal domain.
        """
        return self._domain.send(self._generator_network(latent))


class FrozenGenerator(DNNGenerator):
    """Frozen generator module.

    This module has the generator network as a submodule and its parameters are
    frozen.

    Parameters
    ----------
    generator_network : nn.Module
        Generator network. This network should have a method `forward` that takes
        a latent vector and propagates it through the network.
    domain : Domain, optional
        Domain of the input stimuli to receive. (default: InternalDomain())

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.modules.generator import FrozenGenerator
    >>> generator_network = nn.Sequential(
    ...     nn.ConvTranspose2d(3, 3, 3),
    ...     nn.ReLU(),
    ... )
    >>> generator = FrozenGenerator(generator_network)
    >>> latent = torch.randn(1, 3, 64, 64)
    >>> generated_image = generator(latent)
    >>> generated_image.shape
    torch.Size([1, 3, 66, 66])
    """

    def __init__(
        self,
        generator_network: nn.Module,
        domain: Domain = InternalDomain(),
    ) -> None:
        """Initialize the generator."""
        super().__init__(generator_network, domain=domain, reset_fn=lambda _: None)
        self._generator_network.eval()

    def reset_states(self) -> None:
        """Reset the state of the generator."""
        pass

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an empty iterator."""
        return iter([])


def build_generator(
    generator_network: nn.Module,
    domain: Domain = InternalDomain(),
    reset_fn: Callable[[nn.Module], None] = call_reset_parameters,
    frozen: bool = True,
) -> BaseGenerator:
    """Build a generator module.

    This function builds a generator module from a generator network that takes
    a latent vector as an input and returns an image on its own domain. One
    needs to specify the domain of the generator network.

    Parameters
    ----------
    generator_network : nn.Module
        Generator network. This network should have a method `forward` that takes
        a latent vector and returns an image on its own domain.
    domain : Domain, optional
        Domain of the input images to receive. (default: InternalDomain()).
        One needs to specify the domain that corresponds to the generator
        network's output domain.
    reset_fn : Callable[[nn.Module], None], optional
        Function to reset the parameters of the generator network, by default
        call_reset_parameters.
    frozen : bool, optional
        Whether to freeze the parameters of the generator network, by default True.

    Returns
    -------
    BaseGenerator
        Generator module.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.modules.generator import build_generator
    >>> generator_network = nn.Sequential(
    ...     nn.ConvTranspose2d(3, 3, 3),
    ...     nn.ReLU(),
    ... )
    >>> generator = build_generator(generator_network)
    >>> latent = torch.randn(1, 3, 64, 64)
    >>> generated_image = generator(latent)
    >>> generated_image.shape
    torch.Size([1, 3, 66, 66])
    """
    if frozen:
        return FrozenGenerator(generator_network, domain=domain)
    else:
        return DNNGenerator(generator_network, domain=domain, reset_fn=reset_fn)
