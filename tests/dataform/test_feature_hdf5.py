import gc
import glob
import os
import stat
import tempfile
import unittest
import warnings

import h5py
import numpy as np
from numpy.testing import assert_array_equal

from bdpy.dataform import Features
from bdpy.dataform._feature_store import (
    MatFeatureStore,
    SUPPORTED_FORMAT_VERSION,
    FORMAT_ATTR,
    FORMAT_NAME,
    FORMAT_VERSION_ATTR,
    SUPPORTED_FORMAT_VERSION,
    HDF5FeatureStore,
)
from bdpy.dataform.feature_hdf5 import (
    FeatureWriter,
    convert_features_to_hdf5,
    save_features,
)

from .test_features import prepare_mat_features


def _current_umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


class TestSaveFeatures(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.labels = ['img%04d' % i for i in range(12)]
        self.data = np.random.rand(12, 32, 5, 5).astype(np.float32)
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_writes_schema_v1(self):
        save_features(self.path, self.data, self.labels)

        with h5py.File(self.path, 'r') as f:
            self.assertEqual(f.attrs[FORMAT_ATTR], FORMAT_NAME)
            self.assertEqual(
                int(f.attrs[FORMAT_VERSION_ATTR]), SUPPORTED_FORMAT_VERSION
            )
            assert_array_equal(f['features'][()], self.data)
            self.assertEqual(
                [s.decode('utf-8') for s in f['labels'][()]], self.labels
            )

    def test_features_dataset_is_chunked_and_uncompressed(self):
        # Chunking is the whole point; compression would tax every partial read.
        save_features(self.path, self.data, self.labels)
        with h5py.File(self.path, 'r') as f:
            self.assertIsNotNone(f['features'].chunks)
            self.assertIsNone(f['features'].compression)

    def test_explicit_chunks_and_compression(self):
        save_features(
            self.path, self.data, self.labels,
            chunks=(4, 8, 5, 5), compression='gzip',
        )
        with h5py.File(self.path, 'r') as f:
            self.assertEqual(f['features'].chunks, (4, 8, 5, 5))
            self.assertEqual(f['features'].compression, 'gzip')
        # Compression must not change what comes back out.
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_dtype_is_preserved(self):
        for dtype in (np.float32, np.float64, np.int32):
            with self.subTest(dtype=dtype):
                path = os.path.join(self.tmpdir.name, 'l_%s.h5' % np.dtype(dtype).name)
                save_features(path, self.data.astype(dtype), self.labels)
                with h5py.File(path, 'r') as f:
                    self.assertEqual(f['features'].dtype, np.dtype(dtype))

    def test_rejects_mismatched_labels(self):
        with self.assertRaises(ValueError):
            save_features(self.path, self.data, self.labels[:-1])

    def test_rejects_1d_features(self):
        with self.assertRaises(ValueError):
            save_features(self.path, np.arange(5), ['a'] * 5)


class TestFeatureWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')
        self.labels = ['img%04d' % i for i in range(150)]
        self.data = np.random.rand(150, 16, 3, 3).astype(np.float32)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_append_round_trip(self):
        # 150 samples crosses several resize blocks, which is where an
        # incremental writer would lose data if the resize were wrong.
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            for label, feature in zip(self.labels, self.data):
                writer.append(feature, label)
            self.assertEqual(writer.n_samples, len(self.labels))

        store = HDF5FeatureStore(self.tmpdir.name)
        self.assertEqual(store.labels, self.labels)
        assert_array_equal(store.read('conv5'), self.data)

    def test_append_accepts_leading_sample_axis(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            writer.append(self.data[0][np.newaxis], self.labels[0])
            writer.append(self.data[1], self.labels[1])
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data[:2]
        )

    def test_extend(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            writer.extend(self.data[:40], self.labels[:40])
            writer.extend(self.data[40:], self.labels[40:])
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_writes_valid_header(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            writer.append(self.data[0], self.labels[0])
        with h5py.File(self.path, 'r') as f:
            self.assertEqual(f.attrs[FORMAT_ATTR], FORMAT_NAME)
            self.assertIsNotNone(f['features'].chunks)

    def test_rejects_wrong_shape(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            with self.assertRaises(ValueError):
                writer.extend(np.zeros((2, 8, 3, 3)), ['a', 'b'])

    def test_rejects_label_count_mismatch(self):
        with FeatureWriter(self.path, (16, 3, 3), np.float32) as writer:
            with self.assertRaises(ValueError):
                writer.extend(np.zeros((2, 16, 3, 3)), ['a'])

    def test_write_after_close_raises(self):
        writer = FeatureWriter(self.path, (16, 3, 3), np.float32)
        writer.close()
        writer.close()  # idempotent
        with self.assertRaises(RuntimeError):
            writer.append(self.data[0], self.labels[0])


class TestConvertFeaturesToHDF5(unittest.TestCase):
    """The converter must reproduce exactly what the legacy reader sees."""

    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.matdir = os.path.join(self.tmpdir.name, 'mat')
        self.h5dir = os.path.join(self.tmpdir.name, 'h5')
        os.makedirs(self.matdir)
        self.labels = ['img%04d' % i for i in range(10)]
        self.stacked = prepare_mat_features(
            self.matdir, ['conv5', 'fc8'], self.labels,
            [(1, 16, 3, 3), (1, 50)],
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_converted_features_match_source(self):
        convert_features_to_hdf5(self.matdir, self.h5dir)

        from_mat = Features(self.matdir)
        from_h5 = Features(self.h5dir)
        self.assertEqual(from_h5.layers, from_mat.layers)
        self.assertEqual(from_h5.labels, from_mat.labels)
        for layer in from_mat.layers:
            with self.subTest(layer=layer):
                assert_array_equal(from_h5.get(layer), self.stacked[layer])
                assert_array_equal(from_h5.get(layer), from_mat.get(layer))

    def test_selected_layers_only(self):
        convert_features_to_hdf5(self.matdir, self.h5dir, layers=['fc8'])
        self.assertEqual(Features(self.h5dir).layers, ['fc8'])

    def test_unknown_layer_raises(self):
        with self.assertRaises(KeyError):
            convert_features_to_hdf5(self.matdir, self.h5dir, layers=['nope'])

    def test_existing_file_is_skipped_unless_overwrite(self):
        convert_features_to_hdf5(self.matdir, self.h5dir)
        marker = os.path.join(self.h5dir, 'fc8.h5')
        mtime = os.path.getmtime(marker)
        os.utime(marker, (mtime - 100, mtime - 100))

        convert_features_to_hdf5(self.matdir, self.h5dir)
        self.assertEqual(os.path.getmtime(marker), mtime - 100)

        convert_features_to_hdf5(self.matdir, self.h5dir, overwrite=True)
        self.assertGreater(os.path.getmtime(marker), mtime - 100)

    def test_small_batches_produce_the_same_file(self):
        # Batching is an implementation detail; it must not affect the result.
        convert_features_to_hdf5(self.matdir, self.h5dir, batch_size=3)
        assert_array_equal(Features(self.h5dir).get('conv5'), self.stacked['conv5'])


class TestSchemaValidation(unittest.TestCase):
    """Everything a reader depends on is checked when the file is opened."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')
        save_features(self.path, np.zeros((3, 4)), ['a', 'b', 'c'])

    def tearDown(self):
        self.tmpdir.cleanup()

    def _mutate(self, fn):
        with h5py.File(self.path, 'a') as f:
            fn(f)

    def test_missing_version_is_rejected(self):
        # Previously read as version 0 and accepted, contradicting the docs.
        self._mutate(lambda f: f.attrs.__delitem__(FORMAT_VERSION_ATTR))
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)

    def test_malformed_version_is_rejected(self):
        # Anything that is not an integer scalar, including a float that would
        # round down to a supported version under int().
        for bad in ('v1', 1.5, 1.0, True, np.True_):
            with self.subTest(version=bad):
                self._mutate(
                    lambda f, b=bad: f.attrs.__setitem__(FORMAT_VERSION_ATTR, b)
                )
                with self.assertRaises(RuntimeError):
                    HDF5FeatureStore(self.tmpdir.name)

    def test_version_below_one_is_rejected(self):
        for bad in (0, -1):
            with self.subTest(version=bad):
                self._mutate(lambda f, b=bad: f.attrs.__setitem__(FORMAT_VERSION_ATTR, b))
                with self.assertRaises(RuntimeError):
                    HDF5FeatureStore(self.tmpdir.name)

    def test_future_version_is_rejected(self):
        self._mutate(
            lambda f: f.attrs.__setitem__(
                FORMAT_VERSION_ATTR, SUPPORTED_FORMAT_VERSION + 1
            )
        )
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        self.assertIn('upgrade bdpy', str(ctx.exception))

    def test_row_count_mismatch_is_rejected(self):
        def shrink(f):
            del f['labels']
            f.create_dataset('labels', data=['a', 'b'],
                             dtype=h5py.string_dtype(encoding='utf-8'))
        self._mutate(shrink)
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        self.assertIn('rows', str(ctx.exception))

    def test_wrong_ndim_is_rejected(self):
        def flatten_features(f):
            del f['features']
            f.create_dataset('features', data=np.zeros(3))
        self._mutate(flatten_features)
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)

        save_features(self.path, np.zeros((2, 4)), ['a', 'b'], overwrite=True)

        def widen_labels(f):
            del f['labels']
            f.create_dataset('labels', data=[['a'], ['b']],
                             dtype=h5py.string_dtype(encoding='utf-8'))
        self._mutate(widen_labels)
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)

    def test_duplicate_labels_are_rejected_on_read(self):
        # Labels map to row indices, so a repeat would resolve both rows to the
        # last one and silently drop the first. The .mat layout cannot express
        # duplicates at all, so neither should this one.
        def duplicate(f):
            del f['labels']
            f.create_dataset('labels', data=['a', 'a', 'c'],
                             dtype=h5py.string_dtype(encoding='utf-8'))
        self._mutate(duplicate)
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        self.assertIn('duplicate', str(ctx.exception))


class TestDuplicateLabelsOnWrite(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_save_features_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            save_features(self.path, np.zeros((2, 4)), ['a', 'a'])
        self.assertFalse(os.path.exists(self.path))

    def test_writer_rejects_duplicates_within_one_call(self):
        with FeatureWriter(self.path, (4,), np.float32) as writer:
            with self.assertRaises(ValueError) as ctx:
                writer.extend(np.zeros((2, 4)), ['a', 'a'])
            self.assertIn('this batch', str(ctx.exception))
            writer.abort()

    def test_writer_rejects_duplicates_across_calls(self):
        with FeatureWriter(self.path, (4,), np.float32) as writer:
            writer.append(np.zeros(4), 'a')
            with self.assertRaises(ValueError) as ctx:
                writer.append(np.ones(4), 'a')
            self.assertIn('already written', str(ctx.exception))
            writer.abort()

    def test_rejected_batch_leaves_earlier_rows_intact(self):
        # The check runs before writing, so a refused batch is a no-op.
        with FeatureWriter(self.path, (4,), np.float32) as writer:
            writer.extend(np.ones((2, 4)), ['a', 'b'])
            with self.assertRaises(ValueError):
                writer.extend(np.zeros((2, 4)), ['c', 'a'])
            self.assertEqual(writer.n_samples, 2)
            writer.extend(np.full((1, 4), 7.0), ['c'])
        store = HDF5FeatureStore(self.tmpdir.name)
        self.assertEqual(store.labels, ['a', 'b', 'c'])


class TestZeroSampleLayer(unittest.TestCase):
    """Both write paths agree on an empty layer."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_save_features_accepts_zero_samples(self):
        path = os.path.join(self.tmpdir.name, 'conv5.h5')
        save_features(path, np.empty((0, 8), dtype=np.float32), [])
        store = HDF5FeatureStore(self.tmpdir.name)
        self.assertEqual(store.labels, [])
        self.assertEqual(store.read('conv5').shape, (0, 8))

    def test_writer_accepts_zero_samples(self):
        path = os.path.join(self.tmpdir.name, 'fc8.h5')
        with FeatureWriter(path, (8,), np.float32):
            pass
        self.assertEqual(HDF5FeatureStore(self.tmpdir.name).read('fc8').shape, (0, 8))


class TestAtomicWriteAndOverwrite(unittest.TestCase):
    """A failed write must leave nothing behind, and must never clobber.

    The converter treats an existing <layer>.h5 as finished and skips it, so a
    half-written file published at the target path would make a truncated layer
    permanent across re-runs.
    """

    def setUp(self):
        warnings.simplefilter('ignore', FutureWarning)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.matdir = os.path.join(self.tmpdir.name, 'mat')
        self.h5dir = os.path.join(self.tmpdir.name, 'h5')
        os.makedirs(self.matdir)
        self.labels = ['img%04d' % i for i in range(10)]
        self.stacked = prepare_mat_features(
            self.matdir, ['conv5'], self.labels, [(1, 16, 3, 3)]
        )
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')
        self.data = np.random.rand(10, 8).astype(np.float32)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _leftovers(self, directory):
        return sorted(os.listdir(directory))

    # --- overwrite guard -------------------------------------------------

    def test_save_features_refuses_to_clobber(self):
        save_features(self.path, self.data, self.labels)
        with self.assertRaises(FileExistsError):
            save_features(self.path, self.data * 2, self.labels)
        # The original must be untouched.
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_save_features_overwrite_replaces(self):
        save_features(self.path, self.data, self.labels)
        save_features(self.path, self.data * 2, self.labels, overwrite=True)
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data * 2
        )

    def test_writer_refuses_to_clobber(self):
        save_features(self.path, self.data, self.labels)
        with self.assertRaises(FileExistsError):
            FeatureWriter(self.path, (8,), np.float32)
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_writer_overwrite_replaces(self):
        save_features(self.path, self.data, self.labels)
        with FeatureWriter(self.path, (8,), np.float32, overwrite=True) as w:
            w.extend(self.data * 3, self.labels)
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data * 3
        )

    # --- atomicity -------------------------------------------------------

    def test_target_is_absent_until_close(self):
        writer = FeatureWriter(self.path, (8,), np.float32)
        writer.append(self.data[0], self.labels[0])
        self.assertFalse(os.path.exists(self.path))
        writer.close()
        self.assertTrue(os.path.exists(self.path))

    def test_abort_leaves_nothing(self):
        writer = FeatureWriter(self.path, (8,), np.float32)
        writer.append(self.data[0], self.labels[0])
        writer.abort()
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self._leftovers(self.tmpdir.name), ['mat'])

    def test_context_manager_aborts_on_exception(self):
        with self.assertRaises(ZeroDivisionError):
            with FeatureWriter(self.path, (8,), np.float32) as writer:
                writer.append(self.data[0], self.labels[0])
                raise ZeroDivisionError
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self._leftovers(self.tmpdir.name), ['mat'])

    def test_close_and_abort_are_idempotent(self):
        writer = FeatureWriter(self.path, (8,), np.float32)
        writer.extend(self.data, self.labels)
        writer.close()
        writer.close()
        writer.abort()  # must not delete the published file
        self.assertTrue(os.path.exists(self.path))
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_failed_save_features_leaves_nothing(self):
        # Mismatched labels raise after the target has been prepared.
        with self.assertRaises(ValueError):
            save_features(self.path, self.data, self.labels[:-1])
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self._leftovers(self.tmpdir.name), ['mat'])

    def test_published_file_is_group_readable(self):
        # Staging must not tighten permissions: these files live on shared lab
        # storage, and 0600 would lock collaborators out.
        save_features(self.path, self.data, self.labels)
        mode = stat.S_IMODE(os.stat(self.path).st_mode)
        expected = 0o666 & ~_current_umask()
        self.assertEqual(mode, expected)

    def test_staging_file_is_not_mistaken_for_a_layer(self):
        # A directory being written into is also a directory someone may read.
        # The scratch file must be invisible to the *.h5 scan, or a concurrent
        # reader would either invent a bogus layer or fail validation.
        writer = FeatureWriter(os.path.join(self.h5dir, 'conv5.h5'), (8,), np.float32)
        try:
            writer.append(self.data[0], self.labels[0])
            self.assertEqual(glob.glob(os.path.join(self.h5dir, '*.h5')), [])
            with self.assertRaises(RuntimeError):
                HDF5FeatureStore(self.h5dir)  # "no .h5 feature file found"
        finally:
            writer.abort()

    def test_failed_overwrite_preserves_the_previous_file(self):
        # The old file stays readable for the whole write and is swapped only at
        # the end, so a failure mid-overwrite loses neither old nor new.
        save_features(self.path, self.data, self.labels)
        with self.assertRaises(ZeroDivisionError):
            with FeatureWriter(self.path, (8,), np.float32, overwrite=True) as w:
                w.extend(self.data * 9, self.labels)
                raise ZeroDivisionError
        assert_array_equal(
            HDF5FeatureStore(self.tmpdir.name).read('conv5'), self.data
        )

    def test_dropped_writer_does_not_leak_a_scratch_file(self):
        writer = FeatureWriter(self.path, (8,), np.float32)
        writer.append(self.data[0], self.labels[0])
        del writer
        gc.collect()
        self.assertEqual(self._leftovers(self.tmpdir.name), ['mat'])

    def test_close_after_abort_is_refused(self):
        # Silently doing nothing would let a caller believe it published.
        writer = FeatureWriter(self.path, (8,), np.float32)
        writer.abort()
        with self.assertRaises(RuntimeError):
            writer.close()
        self.assertFalse(os.path.exists(self.path))

    def test_zero_samples_round_trips(self):
        # An empty layer is representable; refusing it would add a failure mode
        # on the success path for something the caller can check themselves.
        with FeatureWriter(self.path, (8,), np.float32):
            pass
        store = HDF5FeatureStore(self.tmpdir.name)
        self.assertEqual(store.labels, [])
        self.assertEqual(store.read('conv5').shape, (0, 8))

    # --- the converter ---------------------------------------------------

    def _fail_on_second_batch(self):
        """Patch MatFeatureStore.read to blow up partway through a layer."""
        original = MatFeatureStore.read
        state = {'calls': 0}

        def flaky(store, layer, labels=None, feature_slice=None):
            state['calls'] += 1
            if state['calls'] > 1:
                raise OSError('simulated read failure')
            return original(store, layer, labels, feature_slice)

        return original, flaky

    def test_non_positive_batch_size_is_refused(self):
        # range(0, n, -1) is empty, so this used to write nothing and then
        # publish the empty file as a finished layer.
        for bad in (-1, 0):
            with self.subTest(batch_size=bad):
                with self.assertRaises(ValueError):
                    convert_features_to_hdf5(
                        self.matdir, self.h5dir, batch_size=bad
                    )
                self.assertFalse(os.path.exists(self.h5dir))

    def test_failed_conversion_leaves_no_file(self):
        original, flaky = self._fail_on_second_batch()
        MatFeatureStore.read = flaky
        try:
            with self.assertRaises(OSError):
                convert_features_to_hdf5(
                    self.matdir, self.h5dir, batch_size=3
                )
        finally:
            MatFeatureStore.read = original

        out_path = os.path.join(self.h5dir, 'conv5.h5')
        self.assertFalse(os.path.exists(out_path))
        # No scratch file left behind either.
        self.assertEqual(self._leftovers(self.h5dir), [])

    def test_rerun_after_failure_succeeds(self):
        # The regression this guards: a published partial file would be taken
        # for a finished one by the skip check and never repaired.
        original, flaky = self._fail_on_second_batch()
        MatFeatureStore.read = flaky
        try:
            with self.assertRaises(OSError):
                convert_features_to_hdf5(
                    self.matdir, self.h5dir, batch_size=3
                )
        finally:
            MatFeatureStore.read = original

        convert_features_to_hdf5(self.matdir, self.h5dir, batch_size=3)
        assert_array_equal(
            Features(self.h5dir).get('conv5'), self.stacked['conv5']
        )


class TestFormatValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'conv5.h5')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_foreign_hdf5_is_rejected(self):
        # A file that merely happens to have /features is not bdpy storage.
        with h5py.File(self.path, 'w') as f:
            f.create_dataset('features', data=np.zeros((2, 3)))
            f.create_dataset('labels', data=['a', 'b'],
                             dtype=h5py.string_dtype(encoding='utf-8'))
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)

    def test_future_version_is_rejected_with_a_clear_message(self):
        save_features(self.path, np.zeros((2, 3)), ['a', 'b'])
        with h5py.File(self.path, 'a') as f:
            f.attrs[FORMAT_VERSION_ATTR] = SUPPORTED_FORMAT_VERSION + 1
        with self.assertRaises(RuntimeError) as ctx:
            HDF5FeatureStore(self.tmpdir.name)
        self.assertIn('upgrade bdpy', str(ctx.exception))

    def test_missing_dataset_is_rejected(self):
        save_features(self.path, np.zeros((2, 3)), ['a', 'b'])
        with h5py.File(self.path, 'a') as f:
            del f['labels']
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)

    def test_empty_directory_is_rejected(self):
        with self.assertRaises(RuntimeError):
            HDF5FeatureStore(self.tmpdir.name)


if __name__ == "__main__":
    unittest.main()
