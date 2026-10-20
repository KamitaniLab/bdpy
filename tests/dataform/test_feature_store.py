"""Backend equivalence for feature storage.

The legacy .mat layout and chunked HDF5 storage must be indistinguishable
through the Features API: same labels, same order, same arrays, for every
combination of label selection and feature slicing. These tests run each query
against both backends and compare.
"""

import os
import tempfile
import unittest
import warnings
from unittest import mock

import h5py
import numpy as np
from numpy.testing import assert_array_equal

from bdpy.dataform import Features, convert_features_to_hdf5, save_features
from bdpy.dataform import _feature_store
from bdpy.dataform._feature_store import (
    FeatureStore,
    HDF5FeatureStore,
    MatFeatureStore,
    _axis_index,
    _check_size,
    _default_iter_size,
    _iter_array_slabs,
    _normalize_axis,
    _normalize_feature_slice,
    _partition,
    detect_format,
)

from .test_features import prepare_mat_features

LAYERS = ['conv5', 'fc8']
SHAPES = [(1, 24, 5, 5), (1, 60)]
LABELS = ['img%04d' % i for i in range(11)]

# Every way a caller might ask for features. Each is run against both backends.
QUERIES = [
    ('all', 'conv5', None, None),
    ('single label', 'conv5', 'img0003', None),
    ('label list', 'conv5', ['img0005', 'img0001', 'img0009'], None),
    ('unsorted labels', 'conv5', ['img0009', 'img0000'], None),
    ('repeated labels', 'conv5', ['img0009', 'img0001', 'img0009', 'img0000'], None),
    ('feature slice', 'conv5', None, np.s_[8:16]),
    ('slice with step', 'conv5', None, np.s_[::2]),
    ('multi-axis slice', 'conv5', None, np.s_[8:16, 1:4, :]),
    ('integer index', 'conv5', None, np.s_[5]),
    ('ellipsis', 'conv5', None, np.s_[..., 1:3]),
    ('labels and slice', 'conv5', ['img0009', 'img0001', 'img0009'], np.s_[8:16]),
    ('2d layer', 'fc8', None, np.s_[10:40]),
    ('2d layer with labels', 'fc8', ['img0002', 'img0000'], np.s_[10:40]),
]


# feature_slice specs outside basic forward indexing. Validation happens once,
# in FeatureStore.read, so the table is checked against the validator itself;
# TestFeatureSliceValidation checks that every backend goes through it.
REJECTED_FEATURE_SLICES = [
    ('negative step', np.s_[::-1]),
    ('reversed with bounds', np.s_[16:4:-1]),
    ('zero step', slice(None, None, 0)),
    ('list', np.s_[[3, 1, 7]]),
    ('ndarray', np.array([1, 2])),
    ('range', range(3)),
    ('bool', True),
    ('numpy bool', np.True_),
    ('newaxis in tuple', (np.newaxis, slice(None))),
    ('two ellipses', (Ellipsis, Ellipsis)),
    ('float start', slice(1.5, 3)),
    ('float stop', slice(1, 3.5)),
    ('float step', slice(None, None, 2.0)),
    ('bool start', slice(True, 3)),
]


class TestIterationHelpers(unittest.TestCase):
    """The index arithmetic behind iter_chunks, on plain in-memory values."""

    def test_normalize_axis(self):
        shape = (11, 24, 5, 5)
        for axis, expected in ((0, 0), (3, 3), (-1, 3), (-4, 0)):
            with self.subTest(axis=axis):
                self.assertEqual(_normalize_axis(axis, shape), expected)

    def test_normalize_axis_rejects_out_of_range(self):
        shape = (11, 24, 5, 5)
        for axis, message in (
            (4, 'axis 4 is out of range for selected shape (11, 24, 5, 5)'),
            (-5, 'axis -5 is out of range for selected shape (11, 24, 5, 5)'),
        ):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError) as ctx:
                    _normalize_axis(axis, shape)
                self.assertEqual(str(ctx.exception), message)

    def test_check_size(self):
        _check_size(1)
        for bad in (0, -1):
            with self.subTest(size=bad):
                with self.assertRaises(ValueError):
                    _check_size(bad)

    def test_partition(self):
        cases = [
            ('exact', 6, 3, [slice(0, 3), slice(3, 6)]),
            ('partial last', 7, 3, [slice(0, 3), slice(3, 6), slice(6, 7)]),
            ('size above length', 4, 10, [slice(0, 4)]),
            ('size one', 3, 1, [slice(0, 1), slice(1, 2), slice(2, 3)]),
            ('empty', 0, 3, []),
        ]
        for name, length, size, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(list(_partition(length, size)), expected)

    def test_axis_index(self):
        sl = slice(2, 4)
        self.assertEqual(_axis_index(0, sl), (sl,))
        self.assertEqual(_axis_index(2, sl), (slice(None), slice(None), sl))

    def test_default_size_follows_the_chunk_extent(self):
        shape = (3, 10, 4)
        self.assertEqual(_default_iter_size(shape, 1, 8, extent=4), 4)
        # Never longer than the axis, never below one.
        self.assertEqual(_default_iter_size(shape, 1, 8, extent=50), 10)
        self.assertEqual(_default_iter_size(shape, 1, 8, extent=0), 1)

    def test_default_size_without_extent_follows_the_byte_budget(self):
        # One element along axis 1 spans 3 * 4 items of 8 bytes = 96 bytes.
        shape = (3, 10, 4)
        with mock.patch.object(_feature_store, 'DEFAULT_TARGET_CHUNK_BYTES', 300):
            self.assertEqual(_default_iter_size(shape, 1, 8, extent=None), 3)
        with mock.patch.object(_feature_store, 'DEFAULT_TARGET_CHUNK_BYTES', 10):
            self.assertEqual(_default_iter_size(shape, 1, 8, extent=None), 1)
        with mock.patch.object(_feature_store, 'DEFAULT_TARGET_CHUNK_BYTES', 10 ** 6):
            self.assertEqual(_default_iter_size(shape, 1, 8, extent=None), 10)

    def test_iter_array_slabs(self):
        array = np.arange(2 * 7 * 3).reshape(2, 7, 3)
        slabs = list(_iter_array_slabs(array, 1, _partition(7, 3)))
        self.assertEqual([sl for sl, _ in slabs], list(_partition(7, 3)))
        for sl, block in slabs:
            assert_array_equal(block, array[:, sl])
        assert_array_equal(np.concatenate([b for _, b in slabs], axis=1), array)

    def test_rejected_feature_slices(self):
        for name, spec in REJECTED_FEATURE_SLICES:
            with self.subTest(index=name):
                with self.assertRaises(ValueError):
                    _normalize_feature_slice(spec)


class _BackendPair(unittest.TestCase):
    """Builds the same features as a .mat tree and as chunked HDF5."""

    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.matdir = os.path.join(self.tmpdir.name, 'mat')
        self.h5dir = os.path.join(self.tmpdir.name, 'h5')
        os.makedirs(self.matdir)
        self.stacked = prepare_mat_features(self.matdir, LAYERS, LABELS, SHAPES)
        convert_features_to_hdf5(self.matdir, self.h5dir)
        self.from_mat = Features(self.matdir)
        self.from_h5 = Features(self.h5dir)

    def tearDown(self):
        self.tmpdir.cleanup()


class TestBackendEquivalence(_BackendPair):
    def test_metadata_matches(self):
        self.assertEqual(self.from_h5.layers, self.from_mat.layers)
        self.assertEqual(self.from_h5.labels, self.from_mat.labels)
        assert_array_equal(self.from_h5.index, self.from_mat.index)
        for layer in LAYERS:
            self.assertEqual(self.from_h5.shape(layer), self.from_mat.shape(layer))

    def test_queries_match_across_backends(self):
        for name, layer, label, feature_slice in QUERIES:
            with self.subTest(query=name):
                from_mat = self.from_mat.get(
                    layer, label=label, feature_slice=feature_slice
                )
                from_h5 = self.from_h5.get(
                    layer, label=label, feature_slice=feature_slice
                )
                self.assertEqual(from_h5.shape, from_mat.shape)
                assert_array_equal(from_h5, from_mat)

    def test_queries_match_ground_truth(self):
        # Both backends agreeing is not enough if both are wrong the same way.
        for name, layer, label, feature_slice in QUERIES:
            with self.subTest(query=name):
                if label is None:
                    rows = slice(None)
                elif isinstance(label, str):
                    rows = [LABELS.index(label)]
                else:
                    rows = [LABELS.index(s) for s in label]
                expected = self.stacked[layer][rows]
                if feature_slice is not None:
                    indexers = (
                        feature_slice
                        if isinstance(feature_slice, tuple)
                        else (feature_slice,)
                    )
                    expected = expected[(slice(None), *indexers)]
                assert_array_equal(
                    self.from_h5.get(layer, label=label, feature_slice=feature_slice),
                    expected,
                )

    def test_statistic_matches(self):
        for statistic in ('mean', 'std', 'std, ddof=0'):
            for layer in LAYERS:
                with self.subTest(statistic=statistic, layer=layer):
                    assert_array_equal(
                        self.from_h5.statistic(statistic, layer=layer),
                        self.from_mat.statistic(statistic, layer=layer),
                    )


class TestMatIterChunksReadsOnce(_BackendPair):
    """The legacy backend must read a feature axis once, not once per slab.

    ``MatFeatureStore.read`` loads every selected stimulus file, so reading a
    slab at a time made a k-slab iteration cost k full reads of the layer
    (measured: 1088 s against 5 s for one read plus in-memory slicing).
    """

    def setUp(self):
        super().setUp()
        self.store = MatFeatureStore(self.matdir)
        self.reads = []
        original = self.store.read

        def spy(*args, **kwargs):
            self.reads.append((args, kwargs))
            return original(*args, **kwargs)

        self.store.read = spy

    def test_feature_axis_reads_the_selection_once(self):
        slab_counts = {}
        for size in (1, 3, 4, 24, 100, None):
            with self.subTest(size=size):
                self.reads.clear()
                blocks = list(self.store.iter_chunks('conv5', axis=1, size=size))
                self.assertEqual(len(self.reads), 1)
                slab_counts[size] = len(blocks)
        # Not vacuous: the slab count really does vary, and the worst case is
        # the 24 reads this test exists to prevent.
        self.assertEqual(slab_counts[1], 24)
        self.assertEqual(slab_counts[100], 1)
        self.assertGreater(len(set(slab_counts.values())), 1)

    def test_sample_axis_still_reads_one_slab_at_a_time(self):
        # Preloading here would destroy the one guarantee the legacy backend
        # can make: a layer larger than memory streams along the sample axis.
        blocks = list(self.store.iter_chunks('conv5', axis=0, size=4))
        self.assertEqual(len(blocks), 3)
        self.assertEqual(len(self.reads), 3)
        for (args, _), (sl, _) in zip(self.reads, blocks):
            self.assertEqual(list(args[1]), LABELS[sl])

    def test_nothing_is_cached_between_calls(self):
        for call in (1, 2):
            with self.subTest(call=call):
                self.reads.clear()
                list(self.store.iter_chunks('conv5', axis=1, size=4))
                self.assertEqual(len(self.reads), 1)

    def test_writing_to_one_block_does_not_disturb_another(self):
        # Slabs are views into one array now, but they partition the axis, so
        # they cannot alias each other.
        blocks = list(self.from_mat.iter_chunks('conv5', axis=1, size=4))
        blocks[0][1][:] = 0
        assert_array_equal(blocks[1][1], self.stacked['conv5'][:, 4:8])

    def test_validation_precedes_selection_read(self):
        cases = (
            ({'axis': 9}, 'axis 9 is out of range for selected shape (11, 24, 5, 5)'),
            ({'axis': -9}, 'axis -9 is out of range for selected shape (11, 24, 5, 5)'),
            ({'size': 0}, 'size must be positive, got 0'),
        )
        for kwargs, message in cases:
            with self.subTest(**kwargs):
                self.reads.clear()
                with self.assertRaises(ValueError) as ctx:
                    list(self.store.iter_chunks('conv5', **kwargs))
                self.assertEqual(str(ctx.exception), message)
                # Reading the selection up front must not outrun validation.
                self.assertEqual(len(self.reads), 0)
                with self.assertRaises(ValueError) as from_h5:
                    list(HDF5FeatureStore(self.h5dir).iter_chunks('conv5', **kwargs))
                self.assertEqual(str(from_h5.exception), message)


class TestIterChunks(_BackendPair):
    def test_sample_axis_reassembles_to_a_full_read(self):
        # The sample axis is iterated by label, not through a feature slice, so
        # it is a separate path from the one test_feature_axis_blocks_match_ground_truth
        # covers.
        for features in (self.from_mat, self.from_h5):
            with self.subTest(store=type(features._Features__stores[0]).__name__):
                blocks = list(features.iter_chunks('conv5', axis=0, size=4))
                joined = np.concatenate([b for _, b in blocks], axis=0)
                assert_array_equal(joined, self.stacked['conv5'])

    def test_blocks_can_be_sliced_by_the_caller(self):
        # iter_chunks deliberately takes no feature_slice; slicing the blocks as
        # they come out is the supported way to combine the two.
        wanted = np.zeros_like(self.stacked['conv5'][:, :, 1:2])
        for sl, block in self.from_h5.iter_chunks('conv5', axis=1, size=6):
            wanted[:, sl] = block[:, :, 1:2]
        assert_array_equal(wanted, self.stacked['conv5'][:, :, 1:2])

    def test_default_size_is_chunk_aligned(self):
        store = HDF5FeatureStore(self.h5dir)
        extent = store.chunk_extent('conv5', 1)
        sizes = {
            sl.stop - sl.start
            for sl, _ in self.from_h5.iter_chunks('conv5', axis=1)
        }
        # Every block but the last is a whole chunk along the axis.
        self.assertLessEqual(max(sizes), max(1, extent))

    def test_multi_directory_validates_like_a_single_one(self):
        # The multi-store path does not go through FeatureStore.iter_chunks, so
        # it used to skip these checks entirely: size=-1 yielded nothing and a
        # bad axis raised IndexError instead of ValueError.
        other = os.path.join(self.tmpdir.name, 'other')
        os.makedirs(other)
        other_labels = ['other%04d' % i for i in range(3)]
        for layer, shape in zip(LAYERS, SHAPES):
            save_features(
                os.path.join(other, layer + '.h5'),
                np.random.rand(len(other_labels), *shape[1:]),
                other_labels,
            )
        spread = Features([self.h5dir, other])
        both = [LABELS[0], other_labels[0]]
        # Spanning two directories is what takes the fallback path.
        self.assertEqual(len(spread.get('conv5', label=both)), 2)

        for kwargs in ({'size': -1}, {'size': 0}, {'axis': 9}, {'axis': -9}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    list(spread.iter_chunks('conv5', label=both, **kwargs))
                with self.assertRaises(ValueError):
                    list(self.from_h5.iter_chunks('conv5', **kwargs))

    def test_feature_axis_blocks_match_ground_truth(self):
        # What each backend owns is cutting a feature-axis slab out of what it
        # reads; the axis and size arithmetic is checked in
        # TestIterationHelpers. Each size leaves a partial last slab, and the
        # labels repeat and are out of order.
        labels = ['img0009', 'img0001', 'img0009']
        cases = [('conv5', 1, 5), ('conv5', 2, 2), ('conv5', 3, 2), ('fc8', 1, 7)]
        for features in (self.from_mat, self.from_h5):
            for layer, axis, size in cases:
                expected = self.stacked[layer][[LABELS.index(s) for s in labels]]
                with self.subTest(store=type(features._Features__stores[0]).__name__,
                                  layer=layer, axis=axis):
                    blocks = list(features.iter_chunks(
                        layer, label=labels, axis=axis, size=size))
                    self.assertEqual(
                        [sl for sl, _ in blocks],
                        list(_partition(expected.shape[axis], size)),
                    )
                    for sl, block in blocks:
                        assert_array_equal(block, expected[_axis_index(axis, sl)])

    def test_multi_directory_default_size_follows_the_byte_budget(self):
        # Spanning two directories takes the in-memory fallback, which used to
        # yield the whole axis as one slab when size was None. It now splits
        # the way a store without on-disk chunking does. The budget is shrunk
        # so that the small fixture spans several slabs.
        other_dir = tempfile.TemporaryDirectory()
        self.addCleanup(other_dir.cleanup)
        other = other_dir.name
        rng = np.random.default_rng(1)
        for layer, shape in zip(LAYERS, SHAPES):
            data = rng.random((1, *shape[1:]))
            save_features(os.path.join(other, layer + '.h5'), data, ['other0000'])
            if layer == 'conv5':
                other_data = data
        spread = Features([self.matdir, other])
        both = [LABELS[0], 'other0000']
        expected = np.concatenate([self.stacked['conv5'][:1], other_data])

        with mock.patch.object(_feature_store, 'DEFAULT_TARGET_CHUNK_BYTES', 1000):
            from_spread = list(spread.iter_chunks('conv5', label=both, axis=1))
            # The .mat store has no chunk extent, so it uses the same budget.
            from_store = list(self.from_mat.iter_chunks('conv5', label=LABELS[:2], axis=1))

        self.assertGreater(len(from_spread), 1)
        self.assertEqual([sl for sl, _ in from_spread], [sl for sl, _ in from_store])
        for sl, block in from_spread:
            assert_array_equal(block, expected[:, sl])

    def test_negative_axis_is_accepted(self):
        by_negative = list(self.from_h5.iter_chunks('conv5', axis=-3, size=4))
        by_positive = list(self.from_h5.iter_chunks('conv5', axis=1, size=4))
        self.assertEqual(len(by_negative), len(by_positive))
        for (sl_a, a), (sl_b, b) in zip(by_negative, by_positive):
            self.assertEqual(sl_a, sl_b)
            assert_array_equal(a, b)


class TestPartialReads(_BackendPair):
    def test_slice_is_pushed_down_to_h5py(self):
        # The point of the format: h5py must receive the narrowed selection, so
        # that only the covering chunks are read, rather than a full read that
        # NumPy then slices.
        seen = []
        original = h5py.Dataset.__getitem__

        def spy(dataset, args):
            seen.append(args)
            return original(dataset, args)

        h5py.Dataset.__getitem__ = spy
        try:
            self.from_h5.get('conv5', feature_slice=np.s_[8:16])
        finally:
            h5py.Dataset.__getitem__ = original

        self.assertEqual(len(seen), 1)
        rows, feature_index = seen[0]
        self.assertEqual(rows, slice(None))
        self.assertEqual(feature_index, slice(8, 16, None))

    def test_labels_are_read_once_and_in_order(self):
        # Duplicated and out-of-order labels must reach h5py as a single
        # strictly increasing index list, which is all h5py accepts.
        seen = []
        original = h5py.Dataset.__getitem__

        def spy(dataset, args):
            seen.append(args)
            return original(dataset, args)

        h5py.Dataset.__getitem__ = spy
        try:
            out = self.from_h5.get('conv5', label=['img0009', 'img0001', 'img0009'])
        finally:
            h5py.Dataset.__getitem__ = original

        self.assertEqual(len(seen), 1)
        rows = seen[0] if not isinstance(seen[0], tuple) else seen[0][0]
        self.assertEqual(rows, [1, 9])
        assert_array_equal(out, self.stacked['conv5'][[9, 1, 9]])

    def test_dataset_is_chunked(self):
        # This fixture is small enough to fit in one chunk, so only the fact
        # that chunking is enabled can be asserted here. That both sliceable
        # axes are actually split is a property of the chunk-shape policy and
        # is tested against a representative large shape in
        # tests/dataform/test_feature_chunking.py.
        with h5py.File(os.path.join(self.h5dir, 'conv5.h5'), 'r') as f:
            self.assertIsNotNone(f['features'].chunks)


class TestFeatureSliceValidation(_BackendPair):
    """feature_slice is basic forward indexing, identically on both backends.

    The point of narrowing it: whatever is accepted must mean the same thing on
    both layouts, and whatever is rejected must be rejected by both. Previously
    a negative step worked on .mat and raised inside h5py, which is exactly the
    divergence this table guards against.
    """

    accepted = [
        ('plain slice', np.s_[4:16]),
        ('negative stop', np.s_[:-1]),
        ('negative start', np.s_[-8:]),
        ('positive step', np.s_[::2]),
        ('integer', np.s_[5]),
        ('ellipsis', np.s_[..., 1:3]),
        ('multi-axis', np.s_[4:16, 1:3]),
        ('open slice', np.s_[:]),
    ]

    def test_accepted_agree_across_backends_and_numpy(self):
        for name, spec in self.accepted:
            with self.subTest(index=name):
                from_mat = self.from_mat.get('conv5', feature_slice=spec)
                from_h5 = self.from_h5.get('conv5', feature_slice=spec)
                indexers = spec if isinstance(spec, tuple) else (spec,)
                expected = self.stacked['conv5'][(slice(None), *indexers)]
                assert_array_equal(from_mat, expected)
                assert_array_equal(from_h5, expected)

    def test_backends_do_not_override_read(self):
        # FeatureStore.read is where feature_slice is validated; a backend that
        # overrode it could skip validation, which no spy on _read would see.
        for store_cls in (MatFeatureStore, HDF5FeatureStore):
            with self.subTest(store=store_cls.__name__):
                self.assertIs(store_cls.read, FeatureStore.read)

    def test_rejected_by_both_backends_before_backend_read(self):
        # The full table of rejected specs is checked in TestIterationHelpers;
        # here each backend only has to route through the shared validation,
        # and reject before the backend's _read runs.
        for store in (MatFeatureStore(self.matdir), HDF5FeatureStore(self.h5dir)):
            with self.subTest(store=type(store).__name__):
                with mock.patch.object(store, '_read') as backend_read:
                    with self.assertRaises(ValueError):
                        store.read('conv5', feature_slice=np.s_[::-1])
                backend_read.assert_not_called()

    def test_bool_is_not_read_as_an_integer(self):
        # isinstance(True, int) is True in Python, so a naive integer check
        # would silently read True as index 1.
        with self.assertRaises(ValueError):
            self.from_h5.get('conv5', feature_slice=True)

    def test_rejection_message_points_at_the_alternative(self):
        with self.assertRaises(ValueError) as ctx:
            self.from_h5.get('conv5', feature_slice=np.s_[[1, 2]])
        self.assertIn('NumPy', str(ctx.exception))

    def test_iter_chunks_takes_no_feature_slice(self):
        for features in (self.from_mat, self.from_h5):
            with self.assertRaises(TypeError):
                list(features.iter_chunks('conv5', feature_slice=np.s_[0:4]))


class TestFeatureSliceAnnotation(unittest.TestCase):
    def test_type_hints_resolve(self):
        # FeatureSlice once contained an "ellipsis" forward reference, which is
        # not a Python name, so any tool resolving annotations blew up.
        import typing

        from bdpy.dataform._feature_store import FeatureStore

        typing.get_type_hints(FeatureStore.read)
        typing.get_type_hints(MatFeatureStore.read)


class TestCrossLayerLabelConsistency(unittest.TestCase):
    """Every layer must hold the same labels in the same order.

    A row index built from one layer is used against every layer's /features,
    so a layer whose labels are ordered differently would make a label lookup
    silently return another stimulus' row. The legacy .mat backend enforces the
    same invariant, and the HDF5 backend must match it.
    """

    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, layer, labels, n_features=8):
        data = np.random.rand(len(labels), n_features)
        save_features(os.path.join(self.tmpdir.name, layer + '.h5'), data, labels)
        return data

    def test_consistent_layers_are_accepted(self):
        self._write('conv5', LABELS)
        self._write('fc8', LABELS)
        self.assertEqual(Features(self.tmpdir.name).labels, LABELS)

    def test_different_order_is_rejected(self):
        self._write('conv5', LABELS)
        self._write('fc8', list(reversed(LABELS)))
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        message = str(ctx.exception)
        self.assertIn('different order', message)
        self.assertIn('fc8', message)

    def test_different_content_is_rejected(self):
        self._write('conv5', LABELS)
        self._write('fc8', ['other%04d' % i for i in range(len(LABELS))])
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        self.assertIn('different labels', str(ctx.exception))

    def test_different_length_is_rejected(self):
        self._write('conv5', LABELS)
        self._write('fc8', LABELS[:-2])
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        self.assertIn('labels', str(ctx.exception))

    def test_features_construction_also_rejects(self):
        # The guard must fire through the public entry point too.
        self._write('conv5', LABELS)
        self._write('fc8', list(reversed(LABELS)))
        with self.assertRaises(RuntimeError):
            Features(self.tmpdir.name)

    def test_legacy_backend_rejects_the_same_way(self):
        # Parity with the .mat layout, which raises when layers disagree.
        matdir = os.path.join(self.tmpdir.name, 'mat')
        os.makedirs(matdir)
        prepare_mat_features(matdir, ['conv5'], LABELS, [(1, 8)])
        prepare_mat_features(matdir, ['fc8'], LABELS[:-2], [(1, 8)])
        with self.assertRaises(RuntimeError):
            Features(matdir)


class TestFormatDetection(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.matdir = os.path.join(self.tmpdir.name, 'mat')
        self.h5dir = os.path.join(self.tmpdir.name, 'h5')
        os.makedirs(self.matdir)
        self.stacked = prepare_mat_features(self.matdir, LAYERS, LABELS, SHAPES)
        convert_features_to_hdf5(self.matdir, self.h5dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_detects_each_layout(self):
        self.assertEqual(detect_format(self.matdir), 'mat')
        self.assertEqual(detect_format(self.h5dir), 'hdf5')

    def test_unrelated_subdirectory_does_not_change_detection(self):
        # Detection keys off real feature files, not the bare existence of a
        # subdirectory, so a stray directory cannot flip a chunked tree to .mat.
        os.makedirs(os.path.join(self.h5dir, 'notes'))
        self.assertEqual(detect_format(self.h5dir), 'hdf5')
        self.assertEqual(Features(self.h5dir).layers, LAYERS)

    def test_both_layouts_present_is_ambiguous(self):
        # Guessing here means silently reading different data than intended.
        for layer in LAYERS:
            os.makedirs(os.path.join(self.h5dir, layer))
            open(os.path.join(self.h5dir, layer, 'x.mat'), 'w').close()
        with self.assertRaises(RuntimeError) as ctx:
            detect_format(self.h5dir)
        self.assertIn('ambiguous', str(ctx.exception))
        with self.assertRaises(RuntimeError):
            Features(self.h5dir)
        # An explicit format resolves it.
        self.assertEqual(Features(self.h5dir, format='hdf5').layers, LAYERS)

    def test_empty_directory_is_rejected(self):
        empty = os.path.join(self.tmpdir.name, 'empty')
        os.makedirs(empty)
        with self.assertRaises(RuntimeError):
            detect_format(empty)

    def test_explicit_format_overrides_detection(self):
        self.assertIsInstance(
            Features(self.h5dir, format='hdf5')._Features__stores[0],
            HDF5FeatureStore,
        )
        self.assertIsInstance(
            Features(self.matdir, format='mat')._Features__stores[0],
            MatFeatureStore,
        )

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            Features(self.h5dir, format='parquet')

    def test_mixed_directories(self):
        # One dpath per layout, read through a single Features.
        h5_only = os.path.join(self.tmpdir.name, 'h5b')
        os.makedirs(h5_only)
        other_labels = ['other%04d' % i for i in range(4)]
        for layer, shape in zip(LAYERS, SHAPES):
            data = np.random.rand(len(other_labels), *shape[1:])
            save_features(
                os.path.join(h5_only, layer + '.h5'), data, other_labels
            )

        features = Features([self.matdir, h5_only])
        self.assertEqual(features.labels, LABELS + other_labels)
        # A query spanning both directories keeps the caller's order.
        got = features.get('conv5', label=['other0002', 'img0003', 'other0000'])
        assert_array_equal(got[1], self.stacked['conv5'][3])
        self.assertEqual(got.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
