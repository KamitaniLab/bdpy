import torch
import unittest
from typing import Iterator
import torch.nn as nn
from functools import partial
from bdpy.recon.torch.modules import latent as latent_module


class DummyLatent(latent_module.BaseLatent):
    def __init__(self):
        self.latent = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))

    def reset_states(self):
        with torch.no_grad():
            self.latent.fill_(0.0)
    
    def parameters(self, recurse):
        return iter(self.latent)
    
    def generate(self):
        return self.latent
        
class TestBaseLatent(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.latent.BaseLatent."""
    def setUp(self):
        self.latent_value_expected = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))
        self.latent_reset_value_expected = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, latent_module.BaseLatent)

    def test_call(self):
        """Test __call__."""
        latent = DummyLatent()
        self.assertTrue(torch.equal(latent(), self.latent_value_expected))

    def test_parameters(self):
        """test parameters"""
        latent = DummyLatent()
        params = latent.parameters(recurse=True)
        self.assertIsInstance(params, Iterator)
    
    def test_reset_states(self):
        """test reset_states"""
        latent = DummyLatent()
        latent.reset_states()
        self.assertTrue(torch.equal(latent(), self.latent_reset_value_expected))

class DummyNNModuleLatent(latent_module.NNModuleLatent):
    def __init__(self):
        super().__init__()
        self.latent = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))

    def reset_states(self):
       with torch.no_grad():
            self.latent.fill_(0.0)
    
    def generate(self):
        return self.latent

class TestNNModuleLatent(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.latent.NNModuleLatent."""
    def setUp(self):
        self.latent_value_expected = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))
        self.latent_reset_value_expected = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, latent_module.NNModuleLatent)

    def test_call(self):
        """Test __call__."""
        latent = DummyNNModuleLatent()
        self.assertTrue(torch.equal(latent(), self.latent_value_expected))

    def test_parameters(self):
        """test parameters"""
        latent = DummyNNModuleLatent()
        params = latent.parameters(recurse=True)
        self.assertIsInstance(params, Iterator)
    
    def test_reset_states(self):
        """test reset_states"""
        latent = DummyNNModuleLatent()
        latent.reset_states()
        self.assertTrue(torch.equal(latent(), self.latent_reset_value_expected))

class DummyArbitraryLatent(latent_module.ArbitraryLatent):
    def parameters(self, recurse):
        return iter(self._latent)

class TestArbitraryLatent(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.latent.ArbitraryLatent."""
    def setUp(self):
        self.latent = DummyArbitraryLatent((1, 3, 64, 64), partial(nn.init.normal_, mean=0, std=1))
        self.latent_shape_expected = (1, 3, 64, 64)
        self.latent_value_expected = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))
        self.latent_reset_value_expected = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, latent_module.ArbitraryLatent)

    def test_call(self):
        """Test __call__."""
        self.assertEqual(self.latent().size(), self.latent_shape_expected)

    def test_parameters(self):
        """test parameters"""
        params = self.latent.parameters(recurse=True)
        self.assertIsInstance(params, Iterator)
    
    def test_reset_states(self):
        """test reset_states"""
        self.latent.reset_states()
        mean = self.latent().mean().item()
        std = self.latent().std().item()
        self.assertAlmostEqual(mean, 0, places=1)
        self.assertAlmostEqual(std, 1, places=1)

if __name__ == '__main__':
    unittest.main()
