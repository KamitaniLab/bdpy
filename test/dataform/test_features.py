import unittest

import os
from glob import glob

import numpy as np
from numpy.testing import assert_array_equal
import scipy.io as sio

from bdpy.dataform.features import Features


class TestDataformFeatures(unittest.TestCase):
    def setUp(self):
        # Loading test data
        # TODO: Introduce mock dataset for testing
        self.feature_dir = '/home/nu/data/contents_shared/ImageNetTest/derivatives/features/caffe/bvlc_alexnet'

        # AlexNet, fc8, all samples
        self.alexnet_fc8_all = np.vstack(
            [
                sio.loadmat(f)['feat']
                for f in sorted(glob(os.path.join(self.feature_dir, 'fc8', '*.mat')))
            ]
        )

        # AlexNet, conv5, all samples
        self.alexnet_conv5_all = np.vstack(
            [
                sio.loadmat(f)['feat']
                for f in sorted(glob(os.path.join(self.feature_dir, 'conv5', '*.mat')))
            ]
        )

    def test_features_get_features(self):
        feat = Features(self.feature_dir)

        assert_array_equal(
            feat.get_features('fc8'),
            self.alexnet_fc8_all
        )
        assert_array_equal(
            feat.get_features('conv5'),
            self.alexnet_conv5_all
        )

    def test_features_get_all(self):
        feat = Features(self.feature_dir)

        assert_array_equal(
            feat.get('fc8'),
            self.alexnet_fc8_all
        )
        assert_array_equal(
            feat.get('conv5'),
            self.alexnet_conv5_all
        )

    def test_features_get_label(self):
        feat = Features(self.feature_dir)

        labels = 'n01443537_22563'
        index = np.array([0])
        assert_array_equal(
            feat.get('fc8', label=labels),
            self.alexnet_fc8_all[index, :]
        )
        assert_array_equal(
            feat.get('conv5', label=labels),
            self.alexnet_conv5_all[index, :]
        )

        labels = ['n01443537_22563', 'n01677366_18182', 'n04572121_3262']
        index = np.array([0, 2, 49])
        assert_array_equal(
            feat.get('fc8', label=labels),
            self.alexnet_fc8_all[index, :]
        )
        assert_array_equal(
            feat.get('conv5', label=labels),
            self.alexnet_conv5_all[index, :]
        )

if __name__ == "__main__":
    unittest.main()
