"""Tests for bdpy.dl.torch.domain.image_domain."""

import unittest
import torch
import numpy as np
import warnings
from bdpy.dl.torch.domain import image_domain as iamge_domain_module

class TestAffineDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.image_domain.AffineDomain"""
    def setUp(self):
        self.center0d = 0.0
        self.center1d = np.random.randn(3)
        self.center2d = np.random.randn(32, 32)
        self.center3d = np.random.randn(3, 32, 32)
        self.scale0d = 1
        self.scale1d = np.random.randn(3)
        self.scale2d = np.random.randn(32, 32)
        self.scale3d = np.random.randn(3, 32, 32)
        self.image = torch.rand((1, 3, 32, 32))

    def test_instantiation(self):
        """Test instantiation."""
        # Succeeds when center and scale are 0-dimensional
        affine_domain = iamge_domain_module.AffineDomain(self.center0d, self.scale0d)
        self.assertIsInstance(affine_domain, iamge_domain_module.AffineDomain)

        # Succeeds when center and scale are 1-dimensional
        affine_domain = iamge_domain_module.AffineDomain(self.center1d, self.scale1d)
        self.assertIsInstance(affine_domain, iamge_domain_module.AffineDomain)

        # Succeeds when center and scale are 3-dimensional
        affine_domain = iamge_domain_module.AffineDomain(self.center3d, self.scale3d)
        self.assertIsInstance(affine_domain, iamge_domain_module.AffineDomain)

        # Failss when the center is neither 1-dimensional nor 3-dimensional
        with self.assertRaises(ValueError):
            iamge_domain_module.AffineDomain(self.center2d, self.scale0d)

        # Failss when the scale is neither 1-dimensional nor 3-dimensional
        with self.assertRaises(ValueError):
            iamge_domain_module.AffineDomain(self.center0d, self.scale2d)
    
    def test_send_and_receive(self):
        """Test send and receive"""
        # when 0d
        affine_domain = iamge_domain_module.AffineDomain(self.center0d, self.scale0d)
        transformed_image = affine_domain.send(self.image)
        center0d = torch.from_numpy(np.array([self.center0d])[np.newaxis, np.newaxis, np.newaxis])
        scale0d = torch.from_numpy(np.array([self.scale0d])[np.newaxis, np.newaxis, np.newaxis])
        expected_transformed_image = (self.image + center0d) / self.scale0d
        torch.testing.assert_close(transformed_image, expected_transformed_image)
        received_image = affine_domain.receive(transformed_image)
        expected_received_image = expected_transformed_image * scale0d - center0d
        torch.testing.assert_close(received_image, expected_received_image)

        # when 1d
        affine_domain = iamge_domain_module.AffineDomain(self.center1d, self.scale1d)
        transformed_image = affine_domain.send(self.image)
        center1d = self.center1d[np.newaxis, :, np.newaxis, np.newaxis]
        scale1d = self.scale1d[np.newaxis, :, np.newaxis, np.newaxis]
        expected_transformed_image = (self.image + center1d) / scale1d
        torch.testing.assert_close(transformed_image, expected_transformed_image)
        received_image = affine_domain.receive(transformed_image)
        expected_received_image = expected_transformed_image * scale1d - center1d
        torch.testing.assert_close(received_image, expected_received_image)

        # when 3d
        affine_domain = iamge_domain_module.AffineDomain(self.center3d, self.scale3d)
        transformed_image = affine_domain.send(self.image)
        center3d = self.center3d[np.newaxis]
        scale3d = self.scale3d[np.newaxis]
        expected_transformed_image = (self.image + center3d) / scale3d
        torch.testing.assert_close(transformed_image, expected_transformed_image)
        received_image = affine_domain.receive(transformed_image)
        expected_received_image = expected_transformed_image * scale3d - center3d
        torch.testing.assert_close(received_image, expected_received_image)

class TestRGBDomain(unittest.TestCase):
    """Tests fot bdpy.dl.torch.domain.image_domain.BGRDomain"""

    def setUp(self):
        self.bgr_image = torch.rand((1, 3, 32, 32))
        self.rgb_image = self.bgr_image[:, [2, 1, 0], ...]

    def test_send(self):
        """Test send"""
        bgr_domain = iamge_domain_module.BGRDomain()
        transformed_image = bgr_domain.send(self.bgr_image)
        torch.testing.assert_close(transformed_image, self.rgb_image)
    
    def test_receive(self):
        """Tests receive"""
        bgr_domain = iamge_domain_module.BGRDomain()
        received_image = bgr_domain.receive(self.rgb_image)
        torch.testing.assert_close(received_image, self.bgr_image)

class TestPILDomainWithExplicitCrop(unittest.TestCase):
    """Tests fot bdpy.dl.torch.domain.image_domain.PILDomainWithExplicitCrop"""
    def setUp(self):
        self.expected_transformed_image = torch.rand((1, 3, 32, 32))
        self.image = self.expected_transformed_image.permute(0, 2, 3, 1) * 255

    def test_send(self):
        """Test send"""
        pdwe_domain = iamge_domain_module.PILDomainWithExplicitCrop()
        transformed_image = pdwe_domain.send(self.image)
        torch.testing.assert_close(transformed_image, self.expected_transformed_image)

    def test_receive(self):
        """Tests receive"""
        pdwe_domain = iamge_domain_module.PILDomainWithExplicitCrop()
        with warnings.catch_warnings(record=True) as w:
            received_image = pdwe_domain.receive(self.expected_transformed_image)
        self.assertTrue(any(isinstance(warn.message, RuntimeWarning) for warn in w))
        torch.testing.assert_close(received_image, self.image)
    

if __name__ == "__main__":
    unittest.main()