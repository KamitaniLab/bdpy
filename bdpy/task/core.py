from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Iterable, Any, TypeVar, Generic

from bdpy.task.callback import CallbackHandler, BaseCallback


_CallbackType = TypeVar("_CallbackType", bound=BaseCallback)


class BaseTask(ABC, Generic[_CallbackType]):
    """Base class for tasks."""

    def __init__(
        self, callbacks: _CallbackType | Iterable[_CallbackType] | None = None
    ) -> None:
        self._callback_handler = CallbackHandler(callbacks)

    @abstractmethod
    def __call__(self, *inputs, **parameters) -> Any:
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
