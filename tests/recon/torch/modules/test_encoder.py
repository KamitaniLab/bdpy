"""Tests for bdpy.recon.torch.modules.encoder."""

import unittest

import torch

from bdpy.recon.torch.modules import encoder as encoder_module


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
