import unittest

from typing import List, Tuple

import os
from glob import glob
import tempfile

import numpy as np
from numpy.testing import assert_array_equal
import scipy.io as sio
import hdf5storage

from bdpy.dataform.features import Features


def _prepare_mock_data(
        tmpdir: str,
        mock_layer_names: List[str],
        mock_image_names: List[str],
        mock_shapes: List[Tuple[int, ...]]
    ) -> None:
    """Prepare mock data for testing."""
    for layer_name, shape in zip(mock_layer_names, mock_shapes):
        os.makedirs(os.path.join(tmpdir, layer_name))
        for image_name in mock_image_names:
            data = np.random.rand(*shape)
            hdf5storage.savemat(
                os.path.join(tmpdir, layer_name, image_name + '.mat'),
                {'feat': data},
                format='5')


class TestDataformFeatures(unittest.TestCase):
    def setUp(self):
        self.mock_layer_names = ['fc8', 'conv5']
        self.mock_image_names = [
            'n01443537_22563',
            'n01443537_22564',
            'n01677366_18182',
            'n01677370_20000',
            'n04572121_3262',
            'n04572121_3263',
            'n04572121_3264'
        ]
        self.mock_shapes = [(1, 1000), (1, 256, 13, 13)]
        self.feature_dir = tempfile.TemporaryDirectory()
        _prepare_mock_data(
            self.feature_dir.name,
            self.mock_layer_names,
            self.mock_image_names,
            self.mock_shapes
        )

        # Loading test data
        # AlexNet, fc8, all samples
        self.alexnet_fc8_all = np.vstack(
            [
                sio.loadmat(f)['feat']
                for f in sorted(glob(os.path.join(self.feature_dir.name, 'fc8', '*.mat')))
            ]
        )

        # AlexNet, conv5, all samples
        self.alexnet_conv5_all = np.vstack(
            [
                sio.loadmat(f)['feat']
                for f in sorted(glob(os.path.join(self.feature_dir.name, 'conv5', '*.mat')))
            ]
        )

    def tearDown(self):
        self.feature_dir.cleanup()

    def test_features_get_features(self):
        feat = Features(self.feature_dir.name)

        assert_array_equal(
            feat.get_features('fc8'),
            self.alexnet_fc8_all
        )
        assert_array_equal(
            feat.get_features('conv5'),
            self.alexnet_conv5_all
        )

    def test_features_get_all(self):
        feat = Features(self.feature_dir.name)

        assert_array_equal(
            feat.get('fc8'),
            self.alexnet_fc8_all
        )
        assert_array_equal(
            feat.get('conv5'),
            self.alexnet_conv5_all
        )

    def test_features_get_label(self):
        feat = Features(self.feature_dir.name)

        label_idx = 0
        labels = self.mock_image_names[label_idx]
        index = np.array([label_idx])
        assert_array_equal(
            feat.get('fc8', label=labels),
            self.alexnet_fc8_all[index, :]
        )
        assert_array_equal(
            feat.get('conv5', label=labels),
            self.alexnet_conv5_all[index, :]
        )

        index = np.array([0, 2, 5])
        labels = [self.mock_image_names[i] for i in index]
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
