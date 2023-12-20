from __future__ import annotations

from typing import Dict, Iterable, Callable

from itertools import chain

import torch

from ..modules import BaseEncoder, BaseGenerator, BaseLatent, BaseCritic
from bdpy.util.callback import CallbackHandler, BaseCallback, unused

FeatureType = Dict[str, torch.Tensor]


def _apply_to_features(fn: Callable[[torch.Tensor], torch.Tensor], features: FeatureType) -> FeatureType:
    return {k: fn(v) for k, v in features.items()}


class FeatureInversionCallback(BaseCallback):
    """Callback for feature inversion pipeline.

    As a design principle, the callback functions must not have any side effects
    on the pipeline results. It should be used only for logging, visualization,
    etc.
    """

    @unused
    def on_iteration_start(self, *, step: int) -> None:
        """Callback on iteration start."""
        pass

    @unused
    def on_image_generated(self, *, step: int, image: torch.Tensor) -> None:
        """Callback on image generated."""
        pass

    @unused
    def on_feature_extracted(self, *, step: int, features: FeatureType) -> None:
        """Callback on feature extracted."""
        pass

    @unused
    def on_layerwise_loss_calculated(self, *, layer_loss: torch.Tensor, layer_name: str) -> None:
        """Callback on layerwise loss calculated."""
        pass

    @unused
    def on_loss_calculated(self, *, step: int, loss: torch.Tensor) -> None:
        """Callback on loss calculated."""
        pass

    @unused
    def on_backward_end(self, *, step: int) -> None:
        """Callback on backward end."""
        pass

    @unused
    def on_optimizer_step(self, *, step: int) -> None:
        """Callback on optimizer step."""
        pass

    @unused
    def on_iteration_end(self, *, step: int) -> None:
        """Called at the end of each iteration."""
        pass


class CUILoggingCallback(FeatureInversionCallback):
    """Callback for logging on CUI.

    Parameters
    ----------
    interval : int, optional
        Logging interval, by default 1. If `interval` is 1, the callback logs
        every iteration.
    total_steps : int, optional
        Total number of iterations, by default -1. If `total_steps` is -1,
        the callback does not show the total number of iterations.
    """

    def __init__(self, interval: int = 1, total_steps: int = -1) -> None:
        self._interval = interval
        self._total_steps = total_steps
        self._loss: int | float = -1

    def _step_str(self, step: int) -> str:
        if self._total_steps > 0:
            return f"{step+1}/{self._total_steps}"
        else:
            return f"{step+1}"

    def on_loss_calculated(self, *, step: int, loss: torch.Tensor) -> None:
        self._loss = loss.item()

    def on_iteration_end(self, *, step: int) -> None:
        if step % self._interval == 0:
            print(f"Step: [{self._step_str(step)}], Loss: {self._loss:.4f}")


class WandBLoggingCallback(FeatureInversionCallback):
    """Callback for logging on Weights & Biases.

    Parameters
    ----------
    run : wandb.sdk.wandb_run.Run
        Run object of Weights & Biases.
    interval : int, optional
        Logging interval, by default 1. If `interval` is 1, the callback logs
        every iteration.
    media_interval : int, optional
        Logging interval for media, by default 1. If `media_interval` is 1,
        the callback logs every iteration.

    Notes
    -----
    TODO: Currently it does not work because the dependency (wandb) is not installed.
    """

    def __init__(self, run: wandb.sdk.wandb_run.Run, interval: int = 1, media_interval: int = 1) -> None:
        self._run = run
        self._interval = interval
        self._media_interval = media_interval
        self._step = 0

    def on_iteration_start(self, *, step: int) -> None:
        # NOTE: We need to store the global step because we cannot access it
        #      in `on_layerwise_loss_calculated` by design.
        self._step = step

    def on_image_generated(self, *, step: int, image: torch.Tensor) -> None:
        if self._step % self._media_interval == 0:
            image = wandb.Image(image)
            self._run.log({"generated_image": image}, step=self._step)

    def on_layerwise_loss_calculated(self, *, layer_loss: torch.Tensor, layer_name: str) -> None:
        if self._step % self._interval == 0:
            self._run.log({f"critic/{layer_name}": layer_loss.item()}, step=self._step)

    def on_loss_calculated(self, *, step: int, loss: torch.Tensor) -> None:
        if self._step % self._interval == 0:
            self._run.log({"loss": loss.item()}, step=self._step)


class FeatureInversionPipeline:
    """Feature inversion pipeline.

    Parameters
    ----------
    encoder : BaseEncoder
        Encoder module.
    generator : BaseGenerator
        Generator module.
    latent : BaseLatent
        Latent variable module.
    critic : BaseCritic
        Critic module.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning rate scheduler, by default None.
    num_iterations : int, optional
        Number of iterations, by default 1.
    callbacks : FeatureInversionCallback | Iterable[FeatureInversionCallback] | None, optional
        Callbacks, by default None.

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
        encoder: BaseEncoder,
        generator: BaseGenerator,
        latent: BaseLatent,
        critic: BaseCritic,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        num_iterations: int = 1,
        callbacks: FeatureInversionCallback | Iterable[FeatureInversionCallback] | None = None,
    ) -> None:
        self._encoder = encoder
        self._generator = generator
        self._latent = latent
        self._critic = critic
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._num_iterations = num_iterations

        self._callback_handler = CallbackHandler(callbacks)

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
        self._callback_handler.fire("on_pipeline_start")
        for step in range(self._num_iterations):
            self._callback_handler.fire("on_iteration_start", step=step)
            self._optimizer.zero_grad()

            latent = self._latent()
            generated_image = self._generator(latent)
            self._callback_handler.fire("on_image_generated", step=step, image=generated_image.detach())

            features = self._encoder(generated_image)
            self._callback_handler.fire("on_feature_extracted", step=step, features=_apply_to_features(lambda x: x.detach(), features))

            loss = self._critic(features, target_features)
            self._callback_handler.fire("on_loss_calculated", step=step, loss=loss.detach())
            loss.backward()
            self._callback_handler.fire("on_backward_end", step=step)

            self._optimizer.step()
            self._callback_handler.fire("on_optimizer_step", step=step)
            if self._scheduler is not None:
                self._scheduler.step()

            self._callback_handler.fire("on_iteration_end", step=step)

        generated_image = self._generator(self._latent()).detach()

        self._callback_handler.fire("on_pipeline_end")
        return generated_image

    def reset_states(self) -> None:
        """Reset the state of the pipeline."""
        self._generator.reset_states()
        self._latent.reset_states()
        self._optimizer = self._optimizer.__class__(
            chain(
                self._generator.parameters(),
                self._latent.parameters(),
            ),
            **self._optimizer.defaults
        )

    def register_callback(self, callback: FeatureInversionCallback) -> None:
        """Register a callback."""
        self._callback_handler.register(callback)