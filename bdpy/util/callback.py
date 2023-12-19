from __future__ import annotations

from typing import Callable, Type, Any, Iterable
from typing_extensions import Annotated, ParamSpec

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
    @wraps(fn)  # NOTE: preserve name, docstring, etc. of the original function
    def _unused(*args: _P.args, **kwargs: _P.kwargs) -> _Unused:
        raise RuntimeError(f"Function {fn} is decorated with @unused and must not be called.")

    # NOTE: change the return type to Unused
    _unused.__annotations__["return"] = _Unused

    return _unused


class BaseCallback:
    @unused
    def on_pipeline_start(self) -> None:
        """Callback on pipeline start."""
        pass

    @unused
    def on_pipeline_end(self) -> None:
        """Callback on pipeline end."""
        pass


class CallbackHandler:
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

    def fire(self, event_type: str, **kwargs) -> None:
        for callback_method in self._registered_functions[event_type]:
            callback_method(**kwargs)

