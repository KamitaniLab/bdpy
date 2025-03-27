"""Tests for bdpy.task.callback."""

from __future__ import annotations

import unittest
from typing import Any, Callable

from bdpy.task import callback as callback_module


# NOTE: setup functions
def setup_fns() -> list[tuple[Callable, tuple[Any, ...], Any]]:
    def f1(input_: Any) -> None:
        pass

    def f2(input_):
        pass

    def f3(a: int, b: int) -> int:
        return a + b

    class F4:
        def __call__(self, input_: Any) -> None:
            pass

    return [
        (f1, (None,), None),
        (f2, (None,), None),
        (f3, (1, 2), 3),
        (F4(), (None,), None),
    ]


def setup_callback_classes():
    class TaskBaseCallback(callback_module.BaseCallback):
        def __init__(self):
            super().__init__(base_class=TaskBaseCallback)

        @callback_module.unused
        def on_some_event(self, input_):
            pass

    class AppendCallback(TaskBaseCallback):
        def __init__(self):
            self._storage = []

        def on_some_event(self, input_):
            self._storage.append(input_)

    return TaskBaseCallback, AppendCallback


class TestUnused(unittest.TestCase):
    """Tests for unused decorator."""

    def test_unused(self):
        """Test unused decorator.

        Unused decorator should change the return type of the decorated function
        to Annotated[None, "unused"]. The decorated function should raise a
        RuntimeError when called.

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
        params = setup_fns()
        for fn, inputs_, output in params:
            self.assertTrue(
                not hasattr(fn, "__annotations__")
                or fn.__annotations__.get("return", None) != callback_module._Unused
            )
            self.assertEqual(fn(*inputs_), output)
            unused_fn = callback_module.unused(fn)
            self.assertTrue(
                hasattr(unused_fn, "__annotations__")
                and unused_fn.__annotations__.get("return", None)
                == callback_module._Unused
            )
            with self.assertRaises(RuntimeError):
                unused_fn(*inputs_)

    def test_is_unused(self):
        params = setup_fns()
        for fn, _, _ in params:
            self.assertFalse(callback_module._is_unused(fn))
            self.assertTrue(callback_module._is_unused(callback_module.unused(fn)))


class TestBaseCallback(unittest.TestCase):
    def setUp(self):
        self.callback = callback_module.BaseCallback()
        self.expected_method_names = {
            "on_task_start",
            "on_task_end",
        }

    def test_instance_methods(self):
        method_names = {
            event_type
            for event_type in dir(self.callback)
            if event_type.startswith("on_")
            and callable(getattr(self.callback, event_type))
        }
        self.assertEqual(method_names, self.expected_method_names)
        for event_type in method_names:
            fn = getattr(self.callback, event_type)
            self.assertRaises(RuntimeError, fn)

    def test_subclass_definition(self):
        TaskBaseCallback, _ = setup_callback_classes()
        callback = TaskBaseCallback()
        expected_method_names = {"on_task_start", "on_some_event", "on_task_end"}
        method_names = {
            event_type
            for event_type in dir(callback)
            if event_type.startswith("on_") and callable(getattr(callback, event_type))
        }

        self.assertEqual(method_names, expected_method_names)
        for event_type in method_names:
            fn = getattr(callback, event_type)
            self.assertRaises(RuntimeError, fn)

    def test_validate_callback(self):
        TaskBaseCallback, AppendCallback = setup_callback_classes()

        class Unrelated(callback_module.BaseCallback):
            """Valid callback object but is not a subclass of TaskBaseCallback"""

            pass

        class HasUnknownEvent(TaskBaseCallback):
            """Having invalid instance method `on_unknown_event` as a subclass of TaskBaseCallback"""

            def on_unknown_event(self):
                pass

        self.assertIsNone(
            callback_module._validate_callback(AppendCallback(), TaskBaseCallback)
        )
        self.assertRaises(
            TypeError, callback_module._validate_callback, Unrelated(), TaskBaseCallback
        )
        self.assertRaises(ValueError, HasUnknownEvent)


class TestCallbackHandler(unittest.TestCase):
    def test_initialization(self):
        _, AppendCallback = setup_callback_classes()
        c1, c2 = AppendCallback(), AppendCallback()

        handler = callback_module.CallbackHandler()
        self.assertListEqual(handler._callbacks, [])
        self.assertDictEqual(handler._registered_functions, {})

        handler = callback_module.CallbackHandler(c1)
        self.assertListEqual(handler._callbacks, [c1])
        self.assertDictEqual(
            handler._registered_functions,
            {"on_some_event": [c1.on_some_event]},
        )

        handler = callback_module.CallbackHandler([c1, c2])
        self.assertListEqual(handler._callbacks, [c1, c2])
        self.assertDictEqual(
            handler._registered_functions,
            {"on_some_event": [c1.on_some_event, c2.on_some_event]},
        )

    def test_register(self):
        handler = callback_module.CallbackHandler()
        _, AppendCallback = setup_callback_classes()
        cb = AppendCallback()

        self.assertListEqual(handler._callbacks, [])
        self.assertDictEqual(handler._registered_functions, {})
        handler.register(cb)
        self.assertListEqual(handler._callbacks, [cb])
        self.assertDictEqual(
            handler._registered_functions,
            {"on_some_event": [cb.on_some_event]},
        )

    def test_fire(self):
        handler = callback_module.CallbackHandler()
        _, AppendCallback = setup_callback_classes()
        cb = AppendCallback()
        handler.register(cb)

        self.assertListEqual(cb._storage, [])

        handler.fire("on_task_start")
        self.assertListEqual(cb._storage, [])

        handler.fire("on_some_event", input_=1)
        self.assertListEqual(cb._storage, [1])

        handler.fire("on_some_event", input_=2)
        self.assertListEqual(cb._storage, [1, 2])

        # NOTE: fire() should only accept keyword arguments
        self.assertRaises(TypeError, handler.fire, "on_some_event", 1, 2)

        handler.fire("on_task_end")
        self.assertListEqual(cb._storage, [1, 2])


if __name__ == "__main__":
    unittest.main()
