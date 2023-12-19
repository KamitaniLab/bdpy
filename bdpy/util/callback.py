from __future__ import annotations

from typing import Callable, Type, Any, Iterable
from typing_extensions import Annotated, ParamSpec, Unpack

from collections import defaultdict
from functools import wraps


_P = ParamSpec("_P")
_Unused = Annotated[None, "unused"]


def _is_unused(fn: Callable) -> bool:
    return_type: Type | None = fn.__annotations__.get("return", None)
    if return_type is None:
        return False
    return return_type == _Unused


def unused(fn: Callable[_P, Any]) -> Callable[_P, _Unused]:
    """Decorate a function to raise an error when called.

    This decorator marks a function as unused and raises an error when called.
    The type of the return value is changed to `Annotated[None, "unused"]`.

    Parameters
    ----------
    fn : Callable
        Function to decorate.

    Returns
    -------
    Callable
        Decorated function.

    Examples
    --------
    >>> @unused
    ... def f(a: int, b: int, c: int = 0) -> int:
    ...     return a + b + c
    ...
    >>> f(1, 2, 3)
    Traceback (most recent call last):
        ...
        RuntimeError: Function <function f at 0x7f3b5e2d2d30> is decorated with @unused and must not be called.
    """

    @wraps(fn)  # NOTE: preserve name, docstring, etc. of the original function
    def _unused(*args: _P.args, **kwargs: _P.kwargs) -> _Unused:
        raise RuntimeError(f"Function {fn} is decorated with @unused and must not be called.")

    # NOTE: change the return type to Unused
    _unused.__annotations__["return"] = _Unused

    return _unused


class BaseCallback:
    """Base class for callbacks.

    Callbacks are used to hook into the pipeline and execute custom functions
    at specific events. Callback functions must be defined as methods of the
    callback classes. The callback functions must be named as "on_<event_type>".
    As a design principle, the callback functions must not have any side effects
    on the pipeline results. It should be used only for logging, visualization,
    etc.
    """

    @unused
    def on_pipeline_start(self) -> None:
        """Callback on pipeline start."""
        pass

    @unused
    def on_pipeline_end(self) -> None:
        """Callback on pipeline end."""
        pass


class CallbackHandler:
    """Callback handler.

    This class manages the callback objects and fires the callback functions
    registered to the event type. The callback functions must be defined as
    methods of the callback classes. The callback functions must be named as
    "on_<event_type>".

    Parameters
    ----------
    callbacks : BaseCallback | Iterable[BaseCallback] | None, optional
        Callbacks to register, by default None

    Examples
    --------
    >>> class Callback(BaseCallback):
    ...     def on_pipeline_start(self):
    ...         print("Pipeline started.")
    ...
    ...     def on_pipeline_end(self):
    ...         print("Pipeline ended.")
    ...
    >>> handler = CallbackHandler(Callback())
    >>> handler.fire("on_pipeline_start")
    Pipeline started.
    >>> handler.fire("on_pipeline_end")
    Pipeline ended.
    """

    _callbacks: list[BaseCallback]
    _registered_functions: defaultdict[str, list[Callable]]

    def __init__(self, callbacks: BaseCallback | Iterable[BaseCallback] | None = None) -> None:
        self._callbacks = []
        self._registered_functions = defaultdict(list)
        if callbacks is not None:
            if isinstance(callbacks, BaseCallback):
                callbacks = [callbacks]
            for callback in callbacks:
                self.register(callback)

    def register(self, callback: BaseCallback) -> None:
        """Register a callback.

        Parameters
        ----------
        callback : BaseCallback
            Callback to register.

        Raises
        ------
        TypeError
            If the callback is not an instance of BaseCallback.
        """
        if not isinstance(callback, BaseCallback):
            raise TypeError(f"Callback must be an instance of BaseCallback, not {type(callback)}.")

        self._callbacks.append(callback)
        for event_type in dir(callback):
            callback_method = getattr(callback, event_type)
            if not callable(callback_method):
                continue
            if _is_unused(callback_method):
                continue
            if event_type.startswith("_"):
                continue
            if event_type.startswith("on_"):
                self._registered_functions[event_type].append(callback_method)
                continue

    def fire(self, event_type: str, **kwargs: Any) -> None:
        """Execute the callback functions registered to the event type.

        Parameters
        ----------
        event_type : str
            Event type to fire, which must start with "on_".
        kwargs : dict[str, Any]
            Keyword arguments to pass to the callback functions.

        Raises
        ------
        KeyError
            If the event type is not registered.
        """
        for callback_method in self._registered_functions[event_type]:
            callback_method(**kwargs)

