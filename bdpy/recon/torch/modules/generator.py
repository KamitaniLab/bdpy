from abc import ABC, abstractmethod

from typing import Callable, Iterator

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from bdpy.dl.torch.stimulus_domain import Domain, image_domain


@torch.no_grad()
def reset_all_parameters(module: nn.Module) -> None:
    """Reset the parameters of the module."""
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        module.reset_parameters()


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
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator network.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Generated image. The generated images must be in the range [0, 1].
        """
        pass


class NNModuleGenerator(BaseGenerator, nn.Module):
    """Generator module uses __call__ method and parameters method of nn.Module."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return nn.Module.parameters(self, recurse=recurse)

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
        self._domain = image_domain.Zero2OneImageDomain()

    def reset_states(self) -> None:
        """Reset the state of the generator."""
        pass

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator network.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Generated image. The generated images must be in the range [0, 1].
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
        Domain of the input images to receive. (default: Zero2OneImageDomain())
    reset_fn : Callable[[nn.Module], None], optional
        Function to reset the parameters of the generator network, by default
        reset_all_parameters.

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
        domain: Domain = image_domain.Zero2OneImageDomain(),
        reset_fn: Callable[[nn.Module], None] = reset_all_parameters,
    ) -> None:
        """Initialize the generator."""
        super().__init__()
        self._generator_network = generator_network
        self._domain = domain
        self._reset_fn = reset_fn

    def reset_states(self) -> None:
        """Reset the state of the generator."""
        self._generator_network.apply(self._reset_fn)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator network.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Generated image. The generated images must be in the range [0, 1].
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
        Domain of the input images to receive. (default: Zero2OneImageDomain())

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
        domain: Domain = image_domain.Zero2OneImageDomain()
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
    domain: Domain = image_domain.Zero2OneImageDomain(),
    reset_fn: Callable[[nn.Module], None] = reset_all_parameters,
    frozen: bool = True,
) -> BaseGenerator:
    """Build a generator module.

    Parameters
    ----------
    generator_network : nn.Module
        Generator network. This network should have a method `forward` that takes
        a latent vector and propagates it through the network.
    domain : Domain, optional
        Domain of the input images to receive. (default: Zero2OneImageDomain())
    reset_fn : Callable[[nn.Module], None], optional
        Function to reset the parameters of the generator network, by default
        reset_all_parameters.
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
