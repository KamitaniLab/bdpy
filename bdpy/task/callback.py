from __future__ import annotations

from typing import Callable, Type, Any, Iterable, TypeVar, Generic
from typing_extensions import Annotated, ParamSpec

from collections import defaultdict
from functools import wraps


_P = ParamSpec("_P")
_Unused = Annotated[None, "unused"]


def _is_unused(fn: Callable) -> bool:
    if not hasattr(fn, "__annotations__"):
        return False
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
        RuntimeError: Function <function f at ...> is decorated with @unused and must not be called.
    """

    @wraps(fn)  # NOTE: preserve name, docstring, etc. of the original function
    def _unused(*args: _P.args, **kwargs: _P.kwargs) -> _Unused:
        raise RuntimeError(
            f"Function {fn} is decorated with @unused and must not be called."
        )

    # NOTE: change the return type to Unused
    _unused.__annotations__["return"] = _Unused

    return _unused


def _validate_callback(callback: BaseCallback, base_class: Type[BaseCallback]) -> None:
    """Validate a callback.

    Parameters
    ----------
    callback : BaseCallback
        Callback to validate.
    base_class : Type[BaseCallback]
        Base class of the callback.

    Raises
    ------
    TypeError
        If the callback is not an instance of the base class.
    ValueError
        If the callback has an event type that is not acceptable.

    Examples
    --------
    >>> class TaskBaseCallback(BaseCallback):
    ...     @unused
    ...     def on_task_start(self):
    ...         pass
    ...
    ...     @unused
    ...     def on_task_end(self):
    ...         pass
    ...
    >>> class SomeTaskCallback(TaskBaseCallback):
    ...     def on_unacceptable_event(self):
    ...         # do something
    ...
    >>> callback = SomeTaskCallback()
    >>> _validate_callback(callback, TaskBaseCallback)
    Traceback (most recent call last):
        ...
        ValueError: on_unacceptable_event is not an acceptable event type. ...
    """

    if not isinstance(callback, base_class):
        raise TypeError(
            f"Callback must be an instance of {base_class}, not {type(callback)}."
        )
    acceptable_events = []
    for event_type in dir(base_class):
        if event_type.startswith("on_") and callable(getattr(base_class, event_type)):
            acceptable_events.append(event_type)
    for event_type in dir(callback):
        if not (
            event_type.startswith("on_") and callable(getattr(callback, event_type))
        ):
            continue
        if event_type not in acceptable_events:
            raise ValueError(
                f"{event_type} is not an acceptable event type. "
                f"Acceptable event types are {acceptable_events}. "
                f"Please refer to the documentation of {base_class.__name__} for the list of acceptable event types."
            )


class BaseCallback:
    """Base class for callbacks.

    Callbacks are used to hook into the task and execute custom functions
    at specific events. Callback functions must be defined as methods of the
    callback classes. The callback functions must be named as "on_<event_type>".
    As a design principle, the callback functions must not have any side effects
    on the task results. It should be used only for logging, visualization, etc.

    For example, the following callback class logs the start and end of the task.

    >>> class Callback(BaseCallback):
    ...     def on_task_start(self):
    ...         print("Task started.")
    ...
    ...     def on_task_end(self):
    ...         print("Task ended.")
    ...
    >>> callback = Callback()
    >>> some_task = SomeTask()  # Initialize a task object
    >>> some_task.register_callback(callback)
    >>> outputs = some_task(inputs)  # Run the task
    Task started.
    Task ended.

    The set of available events that can be hooked into depends on the task.
    See the base class of the corresponding task for the list of all events.
    `@unused` decorator can be used to mark a callback function as unused, so
    that the callback handler does not fire the function.
    """
    def __init__(self, base_class: Type[BaseCallback] | None = None) -> None:
        if base_class is None:
            base_class = BaseCallback
        _validate_callback(self, base_class)

    @unused
    def on_task_start(self) -> None:
        """Callback on task start."""
        pass

    @unused
    def on_task_end(self) -> None:
        """Callback on task end."""
        pass


_CallbackType = TypeVar("_CallbackType", bound=BaseCallback)


class CallbackHandler(Generic[_CallbackType]):
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
    ...     def __init__(self, name):
    ...         self._name = name
    ...
    ...     def on_task_start(self):
    ...         print(f"Task started (name={self._name}).")
    ...
    ...     def on_task_end(self):
    ...         print(f"Task ended (name={self._name}).")
    ...
    >>> handler = CallbackHandler([Callback("A"), Callback("B")])
    >>> handler.fire("on_task_start")
    Task started (name=A).
    Task started (name=B).
    >>> handler.fire("on_task_end")
    Task ended (name=A).
    Task ended (name=B).
    """

    _callbacks: list[_CallbackType]
    _registered_functions: defaultdict[str, list[Callable]]

    def __init__(
        self, callbacks: _CallbackType | Iterable[_CallbackType] | None = None
    ) -> None:
        self._callbacks = []
        self._registered_functions = defaultdict(list)
        if callbacks is not None:
            if not isinstance(callbacks, Iterable):
                callbacks = [callbacks]
            for callback in callbacks:
                self.register(callback)

    def register(self, callback: _CallbackType) -> None:
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
            raise TypeError(
                f"Callback must be an instance of BaseCallback, not {type(callback)}."
            )

        self._callbacks.append(callback)
        for event_type in dir(callback):
            callback_method = getattr(callback, event_type)
            if not callable(callback_method):
                continue
            if not event_type.startswith("on_"):
                continue
            if _is_unused(callback_method):
                continue
            self._registered_functions[event_type].append(callback_method)

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
