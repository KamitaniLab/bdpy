import torch
import unittest
from abc import ABC, abstractmethod
from typing import Iterator
import torch.nn as nn
from bdpy.recon.torch.modules import latent as latent_module


class DummyLatent(latent_module.BaseLatent):
    def __init__(self):
        self.latent = nn.Parameter(torch.tensor([1.0]))

    def reset_states(self):
        self.latent = torch.zeros_like(self.latent)
    
    def parameters(self, recurse):
        return iter([nn.Parameter(self.latent)])
    
    def generate(self):
        return self.latent
        
class TestBaseLatent(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.latent.BaseLatent."""
    def setUp(self):
        self.latent_value_expected = torch.tensor([1.0])

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, latent_module.BaseLatent)

    def test_call(self):
        """Test __call__."""

        latent = DummyLatent()

        self.assertEqual(latent(), self.latent)

    def test_parameters(self):
        """test parameters"""
        latent = DummyLatent()
        params = latent.parameters(recurse=True)

        self.assertIsInstance(params, Iterator)
        self.assertEqual(next(params).item(), 1.0)
    
    def test_reset_states(self):
        """test reset_states"""
        latent = DummyLatent()
        latent.reset_states()
        params = latent.parameters(recurse=True)

        self.assertEqual(next(params).item(), 0.0)






if __name__ == '__main__':
    unittest.main()
