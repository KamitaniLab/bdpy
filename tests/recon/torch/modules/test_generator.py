"""Tests for bdpy.recon.torch.modules.generator."""

import unittest

import copy

import torch
import torch.nn as nn

from bdpy.recon.torch.modules import generator as generator_module


class TestResetAllParameters(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.reset_all_parameters."""

    def test_reset_all_parameters(self):
        """Test reset_all_parameters."""
        pass


class TestBaseGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.BaseGenerator."""

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, generator_module.BaseGenerator)

    def test_call(self):
        """Test __call__."""

        class ReturnAsIsGenerator(generator_module.BaseGenerator):
            def generate(self, latent):
                return latent

            def reset_states(self) -> None:
                pass

            def parameters(self, recurse=True):
                return iter([])

        generator = ReturnAsIsGenerator()
        latent = torch.randn(1, 3, 64, 64)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 3, 64, 64))


class TestNNModuleGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.NNModuleGenerator."""

    def setUp(self):
        """Set up."""

        class LinearGenerator(generator_module.NNModuleGenerator):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def generate(self, latent):
                return self.fc(latent)

            def reset_states(self) -> None:
                self.fc.apply(generator_module.reset_all_parameters)

        self.generator = LinearGenerator()

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, generator_module.NNModuleGenerator)

    def test_call(self):
        """Test __call__."""
        latent = torch.randn(1, 64)
        generated_image = self.generator(latent)
        self.assertEqual(generated_image.shape, (1, 64))

    def test_reset_states(self):
        """Test reset_states."""
        generator_copy = copy.deepcopy(self.generator)
        for p1, p2 in zip(self.generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        self.generator.reset_states()
        for p1, p2 in zip(self.generator.parameters(), generator_copy.parameters()):
            self.assertFalse(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
