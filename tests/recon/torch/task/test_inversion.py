"""Tests for bdpy.recon.torch.task.inversion"""

from __future__ import annotations

import unittest
from unittest.mock import patch, call
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from bdpy.recon.torch.task import inversion as inversion_module
from bdpy.task import callback as callback_module
from bdpy.dl.torch.domain.image_domain import Zero2OneImageDomain
from bdpy.recon.torch.modules import encoder as encoder_module
from bdpy.recon.torch.modules import generator as generator_module
from bdpy.recon.torch.modules import latent as latent_module
from bdpy.recon.torch.modules import critic as critic_module
from bdpy.recon.torch.modules import optimizer as optimizer_module


class DummyFeatureInversionCallback(inversion_module.FeatureInversionCallback):
        def __init__(self, total_steps = 1):
            super().__init__()
            self._total_steps = total_steps
            self._loss = 0

        def _step_str(self, step: int) -> str:
            if self._total_steps > 0:
                return f"{step+1}/{self._total_steps}"
            else:
                return f"{step+1}"

        def on_task_start(self):
            print('task start')

        def on_iteration_start(self, step):
            print(f"Step [{self._step_str(step)}] start")

        def on_image_generated(self, step, image):
            print(f"Step [{self._step_str(step)}], {image.shape}")

        def on_loss_calculated(self, step, loss):
            self._loss = loss.item()

        def on_iteration_end(self, step):
            print(f"Step [{self._step_str(step)}] end")

        def on_task_end(self):
            print('task end')



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

        class HasUnknownEvent(DummyFeatureInversionCallback):
            """Having invalid instance method `on_unknown_event` as a subclass of TaskFeatureInversionCallback"""

            def on_unknown_event(self):
                pass

        self.assertIsNone(
            callback_module._validate_callback(DummyFeatureInversionCallback(), inversion_module.FeatureInversionCallback)
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


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7 * 7 * 3, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class LinearGenerator(generator_module.NNModuleGenerator):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 7 * 7 * 3)

    def generate(self, latent):
        return self.fc(latent)

    def reset_states(self) -> None:
        self.fc.apply(generator_module.call_reset_parameters)


class DummyNNModuleLatent(latent_module.NNModuleLatent):
    def __init__(self, base_latent):
        super().__init__()
        self.latent = nn.Parameter(base_latent)

    def reset_states(self):
       with torch.no_grad():
            self.latent.fill_(0.0)

    def generate(self):
        return self.latent


class TestFeatureInversionTask(unittest.TestCase):
    """Tests for bdpy.recon.torch.task.inversion.FeatureInversionTask"""
    def setUp(self):
        self.init_latent = torch.randn(1, 10)
        self.target_feature =  {
            'fc1': torch.randn(1, 32),
            'fc2': torch.randn(1, 10)
        }
        self.encoder = encoder_module.SimpleEncoder(
            MLP(), ["fc1", "fc2"], domain=Zero2OneImageDomain()
        )
        self.generator = generator_module.DNNGenerator(LinearGenerator())
        self.latent = DummyNNModuleLatent(self.init_latent.clone())
        self.critic = critic_module.MSE()
        # self.optimizer = optim.SGD([self.latent.latent], lr=0.1)
        self.optimizer_factory = optimizer_module.build_optimizer_factory(optim.SGD, lr=0.1)
        self.callbacks = DummyFeatureInversionCallback()

        self.inversion_task = inversion_module.FeatureInversionTask(
            encoder=self.encoder,
            generator=self.generator,
            latent=self.latent,
            critic=self.critic,
            optimizer_factory=self.optimizer_factory,
            callbacks=self.callbacks
        )

    @patch('builtins.print')
    def test_call(self, mock_print):
        """Test __call__."""
        generated_image = self.inversion_task(self.target_feature)
        self.assertTrue(len(self.inversion_task._callback_handler._callbacks) > 0)

        # test for process
        assert isinstance(generated_image, torch.Tensor)
        self.assertEqual(generated_image.shape, (1, 7 * 7 * 3))
        self.assertIsNotNone(self.inversion_task._generator._generator_network.fc.weight.grad)
        self.assertFalse(torch.equal(self.inversion_task._latent.latent, self.init_latent))


        # test for callbacks
        self.assertTrue(self.inversion_task._callback_handler._callbacks[0]._loss > 0 )
        mock_print.assert_has_calls([
            call('task start'),
            call('Step [1/1] start'),
            call('Step [1/1], torch.Size([1, 147])'),
            call('Step [1/1] end'),
            call('task end'),
        ])

    def test_reset_state(self):
        """Test reset_states."""
        generator_copy = copy.deepcopy(self.inversion_task._generator)
        latent_copy = copy.deepcopy(self.inversion_task._latent)
        for p1, p2 in zip(self.inversion_task._generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        torch.testing.assert_close(self.inversion_task._latent.latent, latent_copy.latent)
        self.inversion_task.reset_states()

        for p1, p2 in zip(self.inversion_task._generator.parameters(), generator_copy.parameters()):
            self.assertFalse(torch.equal(p1, p2))
        self.assertFalse(torch.equal(self.inversion_task._latent.latent, latent_copy.latent))


if __name__ == "__main__":
    unittest.main()
