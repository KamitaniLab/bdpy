"""Tests for bdpy.recon.torch.modules.generator."""

import unittest

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import get_model

from bdpy.recon.torch.modules import generator as generator_module


class LinearGenerator(generator_module.NNModuleGenerator):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def generate(self, latent):
        return self.fc(latent)

    def reset_states(self) -> None:
        self.fc.apply(generator_module.call_reset_parameters)


class TestCallResetParameters(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.call_reset_parameters."""
    def setUp(self):
        self.model_ids = [
            "alexnet",
            "efficientnet_b0",
            "fasterrcnn_resnet50_fpn",
            "inception_v3",
            "resnet18",
            "vgg11",
            "vit_b_16",
        ]
        # NOTE: The following modules are excluded from validation because they
        #       initialize their parameters as constants every time.
        self.excluded_modules = [
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
        ]

    def _validate_module(self, module: nn.Module, module_copy: nn.Module, parent_name: str = ""):
        if isinstance(module, tuple(self.excluded_modules)):
            return
        for (name_p1, p1), (_, p2) in zip(module.named_parameters(recurse=False), module_copy.named_parameters(recurse=False)):
            # NOTE: skip parameters that are prbably not randomly initialized
            if "weight" not in name_p1:
                continue
            self.assertFalse(
                torch.equal(p1, p2),
                msg=f"Parameter {parent_name}.{name_p1} does not change after calling reset_parameters."
            )
        for (name_m1, m1), (_, m2) in zip(module.named_children(), module_copy.named_children()):
            self._validate_module(m1, m2, f"{parent_name}.{name_m1}")

    def test_call_reset_parameters(self):
        """Test call_reset_parameters."""
        for model_id in self.model_ids:
            model = get_model(model_id)
            model_copy = copy.deepcopy(model)
            for (name_p1, p1), (_, p2) in zip(model.named_parameters(), model_copy.named_parameters()):
                self.assertTrue(
                    torch.equal(p1, p2),
                    msg=f"Parameter {name_p1} of {model_id} has been changed by deepcopy."
                )
            model.apply(generator_module.call_reset_parameters)
            self._validate_module(model, model_copy, model_id)


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
        self.generator = LinearGenerator()

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, generator_module.NNModuleGenerator)

    def test_call(self):
        """Test __call__."""
        latent = torch.randn(1, 64)
        generated_image = self.generator(latent)
        self.assertEqual(generated_image.shape, (1, 10))
        generated_image.sum().backward()
        self.assertIsNotNone(self.generator.fc.weight.grad)

    def test_reset_states(self):
        """Test reset_states."""
        generator_copy = copy.deepcopy(self.generator)
        for p1, p2 in zip(self.generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        self.generator.reset_states()
        for p1, p2 in zip(self.generator.parameters(), generator_copy.parameters()):
            self.assertFalse(torch.equal(p1, p2))


class TestBareGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.BareGenerator."""

    def test_call(self):
        """Test __call__."""
        generator = generator_module.BareGenerator(activation=torch.sigmoid)
        latent = torch.randn(1, 3, 64, 64)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 3, 64, 64))
        torch.testing.assert_close(generated_image, torch.sigmoid(latent))


class TestDNNGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.DNNGenerator."""
    def test_call(self):
        """Test __call__."""
        generator_network = LinearGenerator()
        generator = generator_module.DNNGenerator(generator_network)
        latent = torch.randn(1, 64)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 10))
        generated_image.sum().backward()
        self.assertIsNotNone(generator_network.fc.weight.grad)

    def test_reset_states(self):
        """Test reset_states."""
        generator = generator_module.DNNGenerator(LinearGenerator())
        generator_copy = copy.deepcopy(generator)
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        generator.reset_states()
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertFalse(torch.equal(p1, p2))


class TestFrozenGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.FrozenGenerator."""
    def test_call(self):
        """Test __call__."""
        generator_network = LinearGenerator()
        generator = generator_module.FrozenGenerator(generator_network)
        latent = torch.randn(1, 64)
        generated_image = generator(latent)
        self.assertEqual(generated_image.shape, (1, 10))
        self.assertRaises(ValueError, optim.SGD, generator.parameters())

    def test_reset_states(self):
        """Test reset_states."""
        generator = generator_module.FrozenGenerator(LinearGenerator())
        generator_copy = copy.deepcopy(generator)
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        generator.reset_states()
        for p1, p2 in zip(generator.parameters(), generator_copy.parameters()):
            self.assertTrue(torch.equal(p1, p2))


class TestBuildGenerator(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.generator.build_generator."""
    def test_build_generator(self):
        """Test build_generator."""
        generator_network = LinearGenerator()
        generator = generator_module.build_generator(generator_network)
        self.assertIsInstance(generator, generator_module.DNNGenerator)
        generator = generator_module.build_generator(generator_network, frozen=True)
        self.assertIsInstance(generator, generator_module.FrozenGenerator)


if __name__ == "__main__":
    unittest.main()
