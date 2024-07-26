from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Iterable, Any, TypeVar, Generic

from bdpy.task.callback import CallbackHandler, BaseCallback


_CallbackType = TypeVar("_CallbackType", bound=BaseCallback)


class BaseTask(ABC, Generic[_CallbackType]):
    """Base class for tasks.

    Parameters
    ----------
    callbacks : BaseCallback | Iterable[BaseCallback] | None
        Callbacks to register. If `None`, no callbacks are registered.

    Attributes
    ----------
    _callback_handler : CallbackHandler
        Callback handler.

    Notes
    -----
    This class is designed to be used as a base class for tasks. The task
    implementation should override the `__call__` method. The actual interface
    of `__call__` depends on the task. For example, the task may take a single
    input and return a single output, or it may take multiple inputs and return
    multiple outputs. The task may also take keyword arguments. Please refer to
    the documentation of the specific task for details.
    """

    _callback_handler: CallbackHandler[_CallbackType]

    def __init__(
        self, callbacks: _CallbackType | Iterable[_CallbackType] | None = None
    ) -> None:
        self._callback_handler = CallbackHandler(callbacks)

    def __call__(self, *inputs, **parameters) -> Any:
        """Run the task."""
        return self.run(*inputs, **parameters)

    @abstractmethod
    def run(self, *inputs, **parameters) -> Any:
        """Run the task."""
        pass

    def register_callback(self, callback: _CallbackType) -> None:
        """Register a callback.

        Parameters
        ----------
        callback : BaseCallback
            Callback to register.
        """
        self._callback_handler.register(callback)
