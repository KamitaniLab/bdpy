import unittest

from typing import List, Tuple

import os
import tempfile

import numpy as np
from numpy.testing import assert_array_equal
import hdf5storage

from bdpy.dataform import _mat_v73
from bdpy.dataform.features import Features, save_feature


def prepare_mat_features(
        tmpdir: str,
        mock_layer_names: List[str],
        mock_image_names: List[str],
        mock_shapes: List[Tuple[int, ...]]
    ) -> dict:
    """Write a legacy per-stimulus feature directory for testing.

    Files are written as MATLAB v7.3 (HDF5) so the loader's h5py path (used for
    NumPy 2.0 compatibility) is exercised. The stacked arrays are returned so
    tests can compare against them without re-reading the files.
    """
    stacked = {}
    for layer_name, shape in zip(mock_layer_names, mock_shapes):
        os.makedirs(os.path.join(tmpdir, layer_name))
        arrays = []
        # Stack in sorted-filename order to match Features.__get_labels, which
        # sorts the feature files when collecting labels.
        for image_name in sorted(mock_image_names):
            data = np.random.rand(*shape)
            hdf5storage.savemat(
                os.path.join(tmpdir, layer_name, image_name + '.mat'),
                {'feat': data},
                format='7.3',
                store_python_metadata=True)
            arrays.append(data)
        stacked[layer_name] = np.vstack(arrays)
    return stacked


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
        stacked = prepare_mat_features(
            self.feature_dir.name,
            self.mock_layer_names,
            self.mock_image_names,
            self.mock_shapes
        )

        # Expected data (samples stacked in sorted-filename order)
        self.alexnet_fc8_all = stacked['fc8']
        self.alexnet_conv5_all = stacked['conv5']

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


class TestFeaturesPartialRead(unittest.TestCase):
    """feature_slice / iter_chunks on the legacy .mat backend.

    The legacy layout cannot read partially, so these only check that the
    result is the same as slicing a full read -- which is what makes the API
    backend-independent.
    """

    def setUp(self):
        self.layers = ['fc8', 'conv5']
        self.labels = [
            'n01443537_22563',
            'n01443537_22564',
            'n01677366_18182',
            'n04572121_3262',
        ]
        self.shapes = [(1, 100), (1, 32, 5, 5)]
        self.feature_dir = tempfile.TemporaryDirectory()
        self.stacked = prepare_mat_features(
            self.feature_dir.name, self.layers, self.labels, self.shapes
        )

    def tearDown(self):
        self.feature_dir.cleanup()

    def test_shape_without_reading(self):
        feat = Features(self.feature_dir.name)
        self.assertEqual(feat.shape('conv5'), self.stacked['conv5'].shape)
        self.assertEqual(feat.shape('fc8'), self.stacked['fc8'].shape)

    def test_feature_slice(self):
        feat = Features(self.feature_dir.name)
        assert_array_equal(
            feat.get('conv5', feature_slice=np.s_[8:16]),
            self.stacked['conv5'][:, 8:16],
        )
        assert_array_equal(
            feat.get('fc8', feature_slice=np.s_[10:50]),
            self.stacked['fc8'][:, 10:50],
        )

    def test_feature_slice_with_labels(self):
        feat = Features(self.feature_dir.name)
        labels = [self.labels[2], self.labels[0]]
        assert_array_equal(
            feat.get('conv5', label=labels, feature_slice=np.s_[8:16]),
            self.stacked['conv5'][[2, 0]][:, 8:16],
        )

    def test_iter_chunks_reassembles(self):
        feat = Features(self.feature_dir.name)
        blocks = list(feat.iter_chunks('conv5', axis=1, size=7))
        assert_array_equal(
            np.concatenate([b for _, b in blocks], axis=1), self.stacked['conv5']
        )

    def test_unsliced_get_still_uses_the_layer_cache(self):
        # The cache holds a whole layer; a sliced read must not replace it.
        feat = Features(self.feature_dir.name)
        cached = feat.get_features('conv5')
        feat.get('conv5', feature_slice=np.s_[0:2])
        self.assertIs(feat.get_features('conv5'), cached)


class TestFeaturesFeatureIndex(unittest.TestCase):
    """Unit-index selection.

    NOTE: the happy path is not covered here. `feature_index` files are struct
    .mat files, which the current reader (_mat_v73.loadmat_key) cannot load --
    it handles dense arrays only, so a struct raises TypeError. That is a
    pre-existing regression from the hdf5storage -> h5py read-path change
    (issue #106), independent of the storage backends, and is left untouched
    here rather than silently changed. Only the unambiguous case is asserted.
    """

    def setUp(self):
        self.labels = ['img0001', 'img0002', 'img0003']
        self.feature_dir = tempfile.TemporaryDirectory()
        self.stacked = prepare_mat_features(
            self.feature_dir.name, ['fc8'], self.labels, [(1, 20)]
        )

    def tearDown(self):
        self.feature_dir.cleanup()

    def _features_with_index(self):
        """Features with a unit index installed.

        NOTE: the index table is injected directly rather than loaded from a
        file. Loading is broken on `dev` (struct .mat files are unreadable by
        _mat_v73.loadmat_key) and fixing that is out of scope here, but the
        guards below must still be covered.
        """
        feat = Features(self.feature_dir.name)
        feat._Features__feat_index_table = {'fc8': np.array([0, 5, 11, 19])}
        return feat

    def test_feature_slice_with_feature_index_is_refused(self):
        # The unit index addresses the flattened FULL feature space, so applying
        # it to an already-sliced array would silently pick the wrong units.
        feat = self._features_with_index()
        with self.assertRaises(ValueError) as ctx:
            feat.get('fc8', feature_slice=np.s_[0:10])
        self.assertIn('feature_index', str(ctx.exception))

        # ... including when labels are also given.
        with self.assertRaises(ValueError):
            feat.get('fc8', label=self.labels[0], feature_slice=np.s_[0:10])

    def test_unsliced_get_still_works_with_feature_index(self):
        # The guard must not break the supported path.
        feat = self._features_with_index()
        assert_array_equal(
            feat.get('fc8'), self.stacked['fc8'][:, [0, 5, 11, 19]]
        )

    def test_iter_chunks_with_feature_index_is_refused(self):
        feat = self._features_with_index()
        with self.assertRaises(ValueError):
            list(feat.iter_chunks('fc8'))

    def test_missing_index_file_raises(self):
        with self.assertRaises(RuntimeError):
            Features(self.feature_dir.name, feature_index='/no/such/file.mat')


class TestSaveFeature(unittest.TestCase):
    def test_save_feature_warns_future(self):
        # save_feature writes a MATLAB-compatible v7.3 .mat file, a path that is
        # deprecated (it becomes bdpy-native plain HDF5 in a future release); it
        # must emit a FutureWarning while still writing a reloadable file.
        feature = np.random.rand(1, 1000)

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertWarns(FutureWarning):
                save_feature(feature, tmpdir, 'fc8', 'n01443537_22563')

            save_file = os.path.join(tmpdir, 'fc8', 'n01443537_22563.mat')
            self.assertTrue(os.path.exists(save_file))

            from_file = _mat_v73.loadmat_key(save_file, 'feat')
            assert_array_equal(feature, from_file)


if __name__ == "__main__":
    unittest.main()
