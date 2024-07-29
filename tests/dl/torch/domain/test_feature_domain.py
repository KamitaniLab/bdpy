"""Tests for bdpy.dl.torch.domain.feature_domain."""

import unittest
import torch
from bdpy.dl.torch.domain import feature_domain as feature_domain_module


class TestMethods(unittest.TestCase):
    def setUp(self):
        self.lnd_tensor = torch.empty((12, 196, 768))
        self.nld_tensor = torch.empty((196, 12, 768))

    def test_lnd2nld(self):
        """test _lnd2nld"""
        self.assertEqual(feature_domain_module._lnd2nld(self.lnd_tensor).shape, self.nld_tensor.shape)

    def test_nld2lnd(self):
        """test _nld2lnd"""
        self.assertEqual(feature_domain_module._nld2lnd(self.nld_tensor).shape, self.lnd_tensor.shape)


class TestArbitraryFeatureKeyDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.feature_domain.ArbitraryFeatureKeyDomain."""
    def setUp(self):
        self.to_internal_mapping = {
            "self_key1": "internal_key1",
            "self_key2": "internal_key2"
        }
        self.to_self_mapping = {
            "internal_key1": "self_key1",
            "internal_key2": "self_key2"
        }
        self.features = {
            "self_key1": 123,
            "self_key2": 456
        }
        self.internal_features = {
            "internal_key1": 123,
            "internal_key2": 456
        }

    def test_send(self):
        """test send"""
        # when both are specified
        domain = feature_domain_module.ArbitraryFeatureKeyDomain(
            to_internal=self.to_internal_mapping,
            to_self=self.to_self_mapping
        )
        self.assertEqual(domain.send(self.features), self.internal_features)

        # when only to_self is specified
        domain = feature_domain_module.ArbitraryFeatureKeyDomain(
            to_self=self.to_self_mapping
        )
        self.assertEqual(domain.send(self.features), self.internal_features)

        # when only to_internal is specified
        domain = feature_domain_module.ArbitraryFeatureKeyDomain(
            to_internal=self.to_internal_mapping
        )
        self.assertEqual(domain.send(self.features), self.internal_features)
    
    def test_receive(self):
        """test receive"""
        # when both are specified
        domain = feature_domain_module.ArbitraryFeatureKeyDomain(
            to_internal=self.to_internal_mapping,
            to_self=self.to_self_mapping
        )
        self.assertEqual(domain.receive(self.internal_features), self.features)

        # when only to_self is specified
        domain = feature_domain_module.ArbitraryFeatureKeyDomain(
            to_self=self.to_self_mapping
        )
        self.assertEqual(domain.receive(self.internal_features), self.features)

        # when only to_internal is specified
        domain = feature_domain_module.ArbitraryFeatureKeyDomain(
            to_internal=self.to_internal_mapping
        )
        self.assertEqual(domain.receive(self.internal_features), self.features)


if __name__ == "__main__":
    unittest.main()