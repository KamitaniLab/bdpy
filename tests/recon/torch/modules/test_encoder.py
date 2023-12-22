"""Tests for bdpy.recon.torch.modules.encoder."""

import unittest

import torch
import torch.nn as nn

from bdpy.dl.torch.domain.image_domain import Zero2OneImageDomain
from bdpy.recon.torch.modules import encoder as encoder_module


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TestBaseEncoder(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.encoder.BaseEncoder."""

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, encoder_module.BaseEncoder)

    def test_call(self):
        """Test __call__."""

        class ReturnAsIsEncoder(encoder_module.BaseEncoder):
            def encode(self, images):
                return {"image": images}

        encoder = ReturnAsIsEncoder()
        images = torch.randn(1, 3, 64, 64)
        features = encoder(images)
        self.assertDictEqual(features, {"image": images})


class TestNNModuleEncoder(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.encoder.NNModuleEncoder."""

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, encoder_module.NNModuleEncoder)

    def test_call(self):
        """Test __call__."""

        class ReturnAsIsEncoder(encoder_module.NNModuleEncoder):
            def __init__(self) -> None:
                super().__init__()
            def encode(self, images):
                return {"image": images}

        encoder = ReturnAsIsEncoder()

        images = torch.randn(1, 3, 64, 64)
        images.requires_grad = True
        features = encoder(images)
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 1)
        self.assertEqual(features["image"].shape, (1, 3, 64, 64))
        features["image"].sum().backward()
        self.assertIsNotNone(images.grad)


class TestSimpleEncoder(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.encoder.SimpleEncoder."""

    def test_call(self):
        """Test __call__."""
        encoder = encoder_module.SimpleEncoder(
            MLP(), ["fc1", "fc2"], domain=Zero2OneImageDomain()
        )
        images = torch.randn(1, 3, 64, 64).clamp(0, 1)
        images.requires_grad = True
        features = encoder(images)
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 2)
        self.assertEqual(features["fc1"].shape, (1, 256))
        self.assertEqual(features["fc2"].shape, (1, 128))
        features["fc2"].sum().backward()
        self.assertIsNotNone(images.grad)


class TestBuildEncoder(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.encoder.build_encoder."""

    def test_build_encoder(self):
        """Test build_encoder."""
        mlp = MLP()
        encoder_from_builder = encoder_module.build_encoder(
            feature_network=mlp,
            layer_names=["fc1", "fc2"],
            domain=Zero2OneImageDomain(),
        )
        encoder = encoder_module.SimpleEncoder(
            mlp, ["fc1", "fc2"], domain=Zero2OneImageDomain()
        )

        images = torch.randn(1, 3, 64, 64).clamp(0, 1)
        features_from_builder = encoder_from_builder(images)
        features = encoder(images)
        self.assertEqual(type(encoder_from_builder), type(encoder))
        self.assertEqual(features_from_builder.keys(), features.keys())
        for key in features_from_builder.keys():
            self.assertTrue(torch.allclose(features_from_builder[key], features[key]))


if __name__ == "__main__":
    unittest.main()
