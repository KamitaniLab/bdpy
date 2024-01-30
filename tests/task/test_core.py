"""Tests for bdpy.task.core."""

from __future__ import annotations

import unittest

from bdpy.task import core as core_module

class MockCallback(core_module.BaseCallback):
    """Mock callback for testing."""
    def __init__(self):
            self._storage = []
    
    def on_some_event(self, input_):
            self._storage.append(input_)

class MockTask(core_module.BaseTask[MockCallback]):
    """Mock task for testing BaseTask."""
    def __call__(self, *inputs, **parameters):
        self._callback_handler.fire("on_some_event", input_=1)
        return inputs, parameters

class TestBaseTask(unittest.TestCase):
    """Tests forbdpy.task.core.BaseTask """
    def setUp(self):
        self.input1 = 1.0
        self.input2 = 2.0
        self.task_name = "reconstruction"

    def test_initialization_without_callbacks(self):
        """Test initialization without callbacks."""
        task = MockTask()
        self.assertIsInstance(task._callback_handler, core_module.CallbackHandler)
        self.assertEqual(len(task._callback_handler._callbacks), 0)

    def test_initialization_with_callbacks(self):
        """Test initialization with callbacks."""
        mock_callback = MockCallback()
        task = MockTask(callbacks=mock_callback)
        self.assertEqual(len(task._callback_handler._callbacks), 1)
        self.assertIn(mock_callback, task._callback_handler._callbacks)

    def test_register_callback(self):
        """Test register_callback method."""
        task = MockTask()
        mock_callback = MockCallback()
        task.register_callback(mock_callback)
        self.assertIn(mock_callback, task._callback_handler._callbacks)

    def test_call(self):
        """Test __call__"""
        mock_callback = MockCallback()
        task = MockTask(callbacks=mock_callback)
        task_inputs, task_parameters = task(self.input1, self.input2, name=self.task_name)
        self.assertEqual(task_inputs, (self.input1, self.input2))
        self.assertEqual(task_parameters["name"], self.task_name)
        self.assertEqual(mock_callback._storage, [1])
