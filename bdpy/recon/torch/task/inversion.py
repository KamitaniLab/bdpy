from __future__ import annotations

from typing import Dict, Iterable, Callable

from itertools import chain

import torch

from ..modules import BaseEncoder, BaseGenerator, BaseLatent, BaseCritic
from bdpy.task import BaseTask
from bdpy.task.callback import BaseCallback, unused, _validate_callback

FeatureType = Dict[str, torch.Tensor]


def _apply_to_features(
    fn: Callable[[torch.Tensor], torch.Tensor], features: FeatureType
) -> FeatureType:
    return {k: fn(v) for k, v in features.items()}


class FeatureInversionCallback(BaseCallback):
    """Callback for feature inversion task.

    As a design principle, the callback functions must not have any side effects
    on the task results. It should be used only for logging, visualization,
    etc. Please refer to `bdpy.util.callback.BaseCallback` for details of the
    usage of callbacks.
    """

    def __init__(self) -> None:
        super().__init__(base_class=FeatureInversionCallback)

    @unused
    def on_task_start(self) -> None:
        """Callback on task start."""
        pass

    @unused
    def on_iteration_start(self, *, step: int) -> None:
        """Callback on iteration start."""
        pass

    @unused
    def on_image_generated(self, *, step: int, image: torch.Tensor) -> None:
        """Callback on image generated."""
        pass

    @unused
    def on_layerwise_loss_calculated(
        self, *, layer_loss: torch.Tensor, layer_name: str
    ) -> None:
        """Callback on layerwise loss calculated."""
        pass

    @unused
    def on_loss_calculated(self, *, step: int, loss: torch.Tensor) -> None:
        """Callback on loss calculated."""
        pass

    @unused
    def on_iteration_end(self, *, step: int) -> None:
        """Called at the end of each iteration."""
        pass

    @unused
    def on_task_end(self) -> None:
        """Callback on task end."""
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
        super().__init__()
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


class FeatureInversionTask(BaseTask):
    """Feature inversion Task.

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
        Callbacks, by default None. Please refer to `bdpy.util.callback.BaseCallback`
        and `bdpy.recon.torch.task.FeatureInversionCallback` for details.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from bdpy.recon.torch.task import FeatureInversionTask
    >>> from bdpy.recon.torch.modules import build_encoder, build_generator, ArbitraryLatent, TargetNormalizedMSE
    >>> encoder = build_encoder(...)
    >>> generator = build_generator(...)
    >>> latent = ArbitraryLatent(...)
    >>> critic = TargetNormalizedMSE(...)
    >>> optimizer = torch.optim.Adam(latent.parameters())
    >>> task = FeatureInversionTask(
    ...     encoder, generator, latent, critic, optimizer, num_iterations=200,
    ... )
    >>> target_features = encoder(target_image)
    >>> reconstructed_image = task(target_features)
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
        callbacks: FeatureInversionCallback
        | Iterable[FeatureInversionCallback]
        | None = None,
    ) -> None:
        super().__init__(callbacks)
        self._encoder = encoder
        self._generator = generator
        self._latent = latent
        self._critic = critic
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._num_iterations = num_iterations

    def run(
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
            Reconstructed images on the libraries internal domain.
        """
        self._callback_handler.fire("on_task_start")
        self.reset_states()
        for step in range(self._num_iterations):
            self._callback_handler.fire("on_iteration_start", step=step)
            self._optimizer.zero_grad()

            latent = self._latent()
            generated_image = self._generator(latent)
            self._callback_handler.fire(
                "on_image_generated", step=step, image=generated_image.clone().detach()
            )

            features = self._encoder(generated_image)

            loss = self._critic(features, target_features)
            self._callback_handler.fire(
                "on_loss_calculated", step=step, loss=loss.clone().detach()
            )
            loss.backward()

            self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()

            self._callback_handler.fire("on_iteration_end", step=step)

        generated_image = self._generator(self._latent()).detach()

        self._callback_handler.fire("on_task_end")
        return generated_image

    def reset_states(self) -> None:
        """Reset the state of the task."""
        self._generator.reset_states()
        self._latent.reset_states()
        self._optimizer = self._optimizer.__class__(
            chain(
                self._generator.parameters(),
                self._latent.parameters(),
            ),
            **self._optimizer.defaults,
        )
