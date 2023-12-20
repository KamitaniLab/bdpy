"""Tests for bdpy.recon.torch.modules.critic."""

import unittest

import torch

from bdpy.recon.torch.modules import critic as critic_module


class TestBaseCritic(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.critic.BaseCritic."""
    def setUp(self):
        self.features = {
            "conv1": torch.tensor([1.0], requires_grad=True),
            "conv2": torch.tensor([2.0], requires_grad=True),
            "conv3": torch.tensor([3.0], requires_grad=True),
        }
        self.target_features = {
            "conv1": torch.tensor([0.0]),
            "conv2": torch.tensor([1.0]),
            "conv3": torch.tensor([2.0]),
        }

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, critic_module.BaseCritic)

    def test_call(self):
        """Test __call__."""
        class ReturnZeroCritic(critic_module.BaseCritic):
            def compare(self, features, target_features):
                return 0.0

        critic = ReturnZeroCritic()
        self.assertEqual(critic(self.features, self.target_features), 0.0)

    def test_loss_computation(self):
        """Test loss computation."""
        class SumCritic(critic_module.BaseCritic):
            def compare(self, features, target_features):
                loss = 0.0
                for feature, target_feature in zip(features.values(), target_features.values()):
                    loss += torch.sum(torch.abs(feature - target_feature))
                return loss

        critic = SumCritic()
        self.assertEqual(critic(self.features, self.target_features), 3.0)

        for feature in self.features.values():
            feature.grad = None
        loss = critic(self.features, self.target_features)
        loss.backward()
        for feature in self.features.values():
            self.assertIsNotNone(feature.grad)
            self.assertEqual(feature.grad, torch.ones_like(feature))


class TestNNModuleCritic(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.critic.NNModuleCritic."""
    def setUp(self):
        self.features = {
            "conv1": torch.tensor([1.0], requires_grad=True),
            "conv2": torch.tensor([2.0], requires_grad=True),
            "conv3": torch.tensor([3.0], requires_grad=True),
        }
        self.target_features = {
            "conv1": torch.tensor([0.0]),
            "conv2": torch.tensor([1.0]),
            "conv3": torch.tensor([2.0]),
        }

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, critic_module.NNModuleCritic)

    def test_call(self):
        """Test __call__."""
        class ReturnZeroCritic(critic_module.NNModuleCritic):
            def compare(self, features, target_features):
                return 0.0

        critic = ReturnZeroCritic()
        self.assertEqual(critic(self.features, self.target_features), 0.0)

    def test_loss_computation(self):
        """Test loss computation."""
        class SumCritic(critic_module.NNModuleCritic):
            def compare(self, features, target_features):
                loss = 0.0
                for feature, target_feature in zip(features.values(), target_features.values()):
                    loss += torch.sum(torch.abs(feature - target_feature))
                return loss

        critic = SumCritic()
        self.assertEqual(critic(self.features, self.target_features), 3.0)

        for feature in self.features.values():
            feature.grad = None
        loss = critic(self.features, self.target_features)
        loss.backward()
        for feature in self.features.values():
            self.assertIsNotNone(feature.grad)
            self.assertEqual(feature.grad, torch.ones_like(feature))
