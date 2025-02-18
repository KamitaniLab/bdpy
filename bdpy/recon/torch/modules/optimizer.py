from __future__ import annotations

from typing import Callable, Iterator, TYPE_CHECKING
from functools import partial
from itertools import chain

if TYPE_CHECKING:
    from torch.nn import Parameter
    import torch.optim as optim
    from ..modules import BaseGenerator, BaseLatent

    _OptimizerFactoryType = Callable[[BaseGenerator, BaseLatent], optim.Optimizer]
    _SchedulerFactoryType = Callable[[optim.Optimizer], optim.lr_scheduler.LRScheduler]
    _GetParamsFnType = Callable[[BaseGenerator, BaseLatent], Iterator[Parameter]]


def build_optimizer_factory(
    optimizer_class: type[optim.Optimizer],
    *,
    get_params_fn: _GetParamsFnType | None = None,
    **kwargs
) -> _OptimizerFactoryType:
    """Build an optimizer factory.

    Parameters
    ----------
    optimizer_class : type
        Optimizer class.
    get_params_fn : Callable[[BaseGenerator, BaseLatent], Iterator[Parameter]] | None
        Custom function to get parameters from the generator and the latent.
        If None, it uses `chain(generator.parameters(), latent.parameters())`.
    kwargs : dict
        Keyword arguments for the optimizer.

    Returns
    -------
    Callable[[BaseGenerator, BaseLatent], optim.Optimizer]
        Optimizer factory.

    Examples
    --------
    >>> from torch.optim import Adam
    >>> from bdpy.recon.torch.modules import build_optimizer_factory
    >>> optimizer_factory = build_optimizer_factory(Adam, lr=1e-3)
    >>> optimizer = optimizer_factory(generator, latent)
    """
    if get_params_fn is None:
        get_params_fn = lambda generator, latent: chain(
            generator.parameters(), latent.parameters()
        )

    def init_fn(generator: BaseGenerator, latent: BaseLatent) -> optim.Optimizer:
        return optimizer_class(get_params_fn(generator, latent), **kwargs)

    return init_fn


def build_scheduler_factory(
    scheduler_class: type[optim.lr_scheduler.LRScheduler], **kwargs
) -> _SchedulerFactoryType:
    """Build a scheduler factory.

    Parameters
    ----------
    scheduler_class : type
        Scheduler class.
    kwargs : dict
        Keyword arguments for the scheduler.

    Returns
    -------
    Callable[[optim.Optimizer], optim.lr_scheduler.LRScheduler]
        Scheduler factory.

    Examples
    --------
    >>> from torch.optim.lr_scheduler import StepLR
    >>> from bdpy.recon.torch.modules import build_scheduler_factory
    >>> scheduler_factory = build_scheduler_factory(StepLR, step_size=100, gamma=0.1)
    >>> scheduler = scheduler_factory(optimizer)
    """
    return partial(scheduler_class, **kwargs)
