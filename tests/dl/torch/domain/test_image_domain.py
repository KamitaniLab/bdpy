"""Tests for bdpy.dl.torch.domain.image_domain."""

import unittest
import torch
import numpy as np
import warnings
from bdpy.dl.torch.domain import image_domain as image_domain_module


class TestAffineDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.image_domain.AffineDomain"""
    def setUp(self):
        self.params = [
            {"center": 0.2, "scale": 0.5, "dim": 0},
            {"center": np.array([0.2, 0.3, 0.4]), "scale": np.array([0.5, 0.6, 0.7]), "dim": 1},
            {"center": np.array([[0.2, 0.3], [0.4, 0.5]]), "scale": np.array([[0.5, 0.6], [0.7, 0.8]]), "dim": 2},
            {"center": np.random.randn(3, 4, 4) * 0.1, "scale": np.random.randn(3, 4, 4) * 0.1, "dim": 3},
        ]

    def test_instantiation(self):
        """Test instantiation."""
        for p in self.params:
            with self.assertRaises(RuntimeError):
                image_domain_module.AffineDomain(p["center"], p["scale"])


class TestScaledDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.image_domain.ScaledDomain"""
    def setUp(self):
        self.scale = 0.5
        self.image = torch.from_numpy(np.random.randn(1, 3, 4, 4))

    def test_instantiation(self):
        """Test instantiation."""
        scaled_domain = image_domain_module.ScaledDomain(self.scale)
        self.assertIsInstance(scaled_domain, image_domain_module.ScaledDomain)

    def test_send_and_receive(self):
        """Test send and receive"""
        scaled_domain = image_domain_module.ScaledDomain(self.scale)
        scaled_image = scaled_domain.send(self.image)
        torch.testing.assert_close(scaled_image, self.image / self.scale)
        received_image = scaled_domain.receive(scaled_image)
        torch.testing.assert_close(received_image, self.image)


class TestStandardizedDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.image_domain.StandardizedDomain"""
    def setUp(self):
        self.params = [
            {"center": 0.2, "scale": 0.5, "dim": 0},
            {"center": np.array([0.2, 0.3, 0.4]), "scale": np.array([0.5, 0.6, 0.7]), "dim": 1},
            {"center": np.array([[0.2, 0.3], [0.4, 0.5]]), "scale": np.array([[0.5, 0.6], [0.7, 0.8]]), "dim": 2},
            {"center": np.random.randn(3, 4, 4) * 0.1, "scale": np.random.randn(3, 4, 4) * 0.1, "dim": 3},
        ]
        self.image = torch.from_numpy(np.random.randn(1, 3, 4, 4))

    def test_instantiation(self):
        """Test instantiation."""
        for p in self.params:
            if p["dim"] != 2:
                standardized_domain = image_domain_module.StandardizedDomain(p["center"], p["scale"])
                self.assertIsInstance(standardized_domain, image_domain_module.StandardizedDomain)
            else:
                with self.assertRaises(ValueError):
                    image_domain_module.StandardizedDomain(p["center"], p["scale"])

        # Fails when the center is neither 1-dimensional nor 3-dimensional
        with self.assertRaises(ValueError):
            image_domain_module.StandardizedDomain(self.params[2]["center"], self.params[0]["scale"])

        # Fails when the scale is neither 1-dimensional nor 3-dimensional
        with self.assertRaises(ValueError):
            image_domain_module.StandardizedDomain(self.params[0]["center"], self.params[2]["scale"])

    def test_send_and_receive(self):
        """Test send and receive"""
        ## Case: 0d
        center, scale = self.params[0]["center"], self.params[0]["scale"]
        center_pt = torch.from_numpy(np.array([center])[np.newaxis, np.newaxis, np.newaxis])
        scale_pt = torch.from_numpy(np.array([scale])[np.newaxis, np.newaxis, np.newaxis])
        standardized_domain = image_domain_module.StandardizedDomain(center, scale)
        destandardized_image = standardized_domain.send(self.image)
        expected_destandardized_image = self.image * scale_pt + center_pt
        torch.testing.assert_close(destandardized_image, expected_destandardized_image)
        standardized_image = standardized_domain.receive(destandardized_image)
        torch.testing.assert_close(standardized_image, self.image)

        ## Case: 1d
        center, scale = self.params[1]["center"], self.params[1]["scale"]
        center_pt = torch.from_numpy(center[np.newaxis, :, np.newaxis, np.newaxis])
        scale_pt = torch.from_numpy(scale[np.newaxis, :, np.newaxis, np.newaxis])
        standardized_domain = image_domain_module.StandardizedDomain(center, scale)
        destandardized_image = standardized_domain.send(self.image)
        expected_destandardized_image = self.image * scale_pt + center_pt
        torch.testing.assert_close(destandardized_image, expected_destandardized_image)
        standardized_image = standardized_domain.receive(destandardized_image)
        torch.testing.assert_close(standardized_image, self.image)

        ## Case: 3d
        center, scale = self.params[3]["center"], self.params[3]["scale"]
        center_pt = torch.from_numpy(center[np.newaxis])
        scale_pt = torch.from_numpy(scale[np.newaxis])
        standardized_domain = image_domain_module.StandardizedDomain(center, scale)
        destandardized_image = standardized_domain.send(self.image)
        expected_destandardized_image = self.image * scale_pt + center_pt
        torch.testing.assert_close(destandardized_image, expected_destandardized_image)
        standardized_image = standardized_domain.receive(destandardized_image)
        torch.testing.assert_close(standardized_image, self.image)


class TestRGBDomain(unittest.TestCase):
    """Tests fot bdpy.dl.torch.domain.image_domain.BGRDomain"""

    def setUp(self):
        self.bgr_image = torch.rand((1, 3, 32, 32))
        self.rgb_image = self.bgr_image[:, [2, 1, 0], ...]

    def test_send(self):
        """Test send"""
        bgr_domain = image_domain_module.BGRDomain()
        transformed_image = bgr_domain.send(self.bgr_image)
        torch.testing.assert_close(transformed_image, self.rgb_image)

    def test_receive(self):
        """Tests receive"""
        bgr_domain = image_domain_module.BGRDomain()
        received_image = bgr_domain.receive(self.rgb_image)
        torch.testing.assert_close(received_image, self.bgr_image)


class TestPILDomainWithExplicitCrop(unittest.TestCase):
    """Tests fot bdpy.dl.torch.domain.image_domain.PILDomainWithExplicitCrop"""
    def setUp(self):
        self.expected_transformed_image = torch.rand((1, 3, 32, 32))
        self.image = self.expected_transformed_image.permute(0, 2, 3, 1) * 255

    def test_send(self):
        """Test send"""
        pdwe_domain = image_domain_module.PILDomainWithExplicitCrop()
        transformed_image = pdwe_domain.send(self.image)
        torch.testing.assert_close(transformed_image, self.expected_transformed_image)

    def test_receive(self):
        """Tests receive"""
        pdwe_domain = image_domain_module.PILDomainWithExplicitCrop()
        with warnings.catch_warnings(record=True) as w:
            received_image = pdwe_domain.receive(self.expected_transformed_image)
        self.assertTrue(any(isinstance(warn.message, RuntimeWarning) for warn in w))
        torch.testing.assert_close(received_image, self.image)


class TestFixedResolutionDomain(unittest.TestCase):
    """Tests fot bdpy.dl.torch.domain.image_domain.FixedResolutionDomain"""
    def setUp(self):
        self.expected_received_image_size = (1, 3, 16, 16)
        self.image =torch.rand((1, 3, 32, 32))

    def test_send(self):
        """Test send"""
        fr_domain = image_domain_module.FixedResolutionDomain((16, 16))
        with self.assertRaises(RuntimeError):
            fr_domain.send(self.image)

    def test_receive(self):
        """Tests receive"""
        fr_domain = image_domain_module.FixedResolutionDomain((16, 16))

        received_image = fr_domain.receive(self.image)
        self.assertEqual(received_image.size(), self.expected_received_image_size)


if __name__ == "__main__":
    unittest.main()