"""Tests for bdpy.recon.torch.task.inversion"""

from __future__ import annotations

import unittest
from unittest.mock import patch
import torch

from bdpy.recon.torch.task import inversion as inversion_module
from bdpy.task import callback as callback_module


class TaskFeatureInversionCallback(inversion_module.FeatureInversionCallback):
        def __init__(self):
            super().__init__()

        def on_task_start(self):
            print('task start')

class TestFeatureInversionCallback(unittest.TestCase):
    """Tests for bdpy.recon.torch.task.inversion.FeatureInversionCallback"""
    def setUp(self):
        self.callback = inversion_module.FeatureInversionCallback()
        self.expected_method_names = {
            "on_task_start",
            "on_iteration_start",
            "on_image_generated",
            "on_layerwise_loss_calculated",
            "on_loss_calculated",
            "on_iteration_end",
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
    

    def test_validate_callback(self):

        class Unrelated(callback_module.BaseCallback):
            """Valid callback object but is not a subclass of TaskFeatureInversionCallback"""

            pass

        class HasUnknownEvent(TaskFeatureInversionCallback):
            """Having invalid instance method `on_unknown_event` as a subclass of TaskFeatureInversionCallback"""

            def on_unknown_event(self):
                pass

        self.assertIsNone(
            callback_module._validate_callback(TaskFeatureInversionCallback(), inversion_module.FeatureInversionCallback)
        )
        self.assertRaises(
            TypeError, callback_module._validate_callback, Unrelated(), inversion_module.FeatureInversionCallback
        )
        self.assertRaises(ValueError, HasUnknownEvent)


class TestCUILoggingCallback(unittest.TestCase):
    """Tests for bdpy.recon.torch.task.inversion.CUILoggingCallback"""
    def setUp(self):
        self.callback = inversion_module.CUILoggingCallback()
        self.expected_loss = torch.tensor([1.0])
    
    def test_on_loss_culculated(self):
        self.callback.on_loss_calculated(step=0, loss=self.expected_loss)
        self.assertEqual(self.callback._loss, self.expected_loss.item())
    
    @patch('builtins.print')
    def test_on_iteration_end(self, mock_print):
        self.callback.on_iteration_end(step=0)
        mock_print.assert_called_once_with("Step: [1], Loss: -1.0000")

class TestFeatureInversionTask(unittest.TestCase):
    """Tests for bdpy.recon.torch.task.inversion.FeatureInversionTask"""
    def setUp(self):
        pass

    



    


if __name__ == "__main__":
    unittest.main()