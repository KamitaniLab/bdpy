from typing import Dict

from itertools import chain

import torch

from ..interface import Encoder, Generator, Latent, Critic

FeatureType = Dict[str, torch.Tensor]


class FeatureInversionPipeline:
    """Feature inversion pipeline.

    Parameters
    ----------
    encoder : Encoder
        Encoder module.
    generator : Generator
        Generator module.
    latent : Latent
        Latent variable module.
    critic : Critic
        Critic module.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning rate scheduler, by default None.
    num_iterations : int, optional
        Number of iterations, by default 1.
    log_interval : int, optional
        Log interval, by default -1. If -1, logging is disabled.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.pipeline import FeatureInversionPipeline
    >>> from bdpy.recon.torch.modules import build_encoder, build_generator, ArbitraryLatent, TargetNormalizedMSE
    >>> encoder = build_encoder(...)
    >>> generator = build_generator(...)
    >>> latent = ArbitraryLatent(...)
    >>> critic = TargetNormalizedMSE(...)
    >>> optimizer = torch.optim.Adam(latent.parameters())
    >>> pipeline = FeatureInversionPipeline(
    ...     encoder, generator, latent, critic, optimizer
    ... )
    >>> target_features = encoder(target_image)
    >>> pipeline.reset_state()
    >>> reconstructed_image = pipeline(target_features)
    """

    def __init__(
        self,
        encoder: Encoder,
        generator: Generator,
        latent: Latent,
        critic: Critic,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        num_iterations: int = 1,
        log_interval: int = -1,
    ) -> None:
        self._encoder = encoder
        self._generator = generator
        self._latent = latent
        self._critic = critic
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._num_iterations = num_iterations
        self._log_interval = log_interval

    def __call__(
        self,
        target_features: FeatureType,
    ) -> torch.Tensor:
        """Run feature inversion given target features.

        Parameters
        ----------
        target_features : FeatureType
            Target features.

        Returns
        -------
        torch.Tensor
            Reconstructed images which have the similar features to the target features.
        """
        for step in range(self._num_iterations):
            self._optimizer.zero_grad()

            latent = self._latent()
            generated_image = self._generator(latent)

            features = self._encoder(generated_image)

            loss = self._critic(features, target_features)
            loss.backward()

            self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()

            if self._log_interval > 0 and step % self._log_interval == 0:
                print(f"Step: [{step+1}/{self._num_iterations}], Loss: {loss.item():.4f}")

        return self._generator(self._latent()).detach()

    def reset_state(self) -> None:
        """Reset the state of the pipeline."""
        self._generator.reset_state()
        self._latent.reset_state()
        self._optimizer = self._optimizer.__class__(
            chain(
                self._generator.parameters(),
                self._latent.parameters(),
            ),
            **self._optimizer.defaults
        )
