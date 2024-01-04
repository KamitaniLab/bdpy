"""Tests for bdpy.task.callback."""

from __future__ import annotations

import unittest
from typing import Any, Callable

from bdpy.task import callback


# NOTE: setup functions
def setup_fns() -> list[tuple[Callable, tuple[Any], Any]]:

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
            self.assertFalse(callback._is_unused(fn))
            self.assertEqual(fn(*inputs_), output)
            self.assertTrue(callback._is_unused(callback.unused(fn)))
            with self.assertRaises(RuntimeError):
                callback.unused(fn)(*inputs_)

