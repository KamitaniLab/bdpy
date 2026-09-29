import unittest

import numpy as np

from bdpy.dataform._feature_chunking import (
    DEFAULT_TARGET_CHUNK_BYTES,
    choose_chunk_shape,
)


def _nbytes(chunk, dtype):
    return int(np.prod(chunk)) * np.dtype(dtype).itemsize


class TestChooseChunkShape(unittest.TestCase):
    """Chunk shape decides partial-read cost, so pin the policy down."""

    shapes = [
        (1200, 1000),
        (1200, 256, 13, 13),
        (50, 1000),
        (1, 256, 13, 13),
        (3, 4096, 7, 7),
        (1000000, 1),
        (2, 1000000),
        (17, 63, 5),
    ]

    def test_invariants(self):
        for dtype in (np.float32, np.float64, np.int16):
            for shape in self.shapes:
                chunk = choose_chunk_shape(shape, np.dtype(dtype))
                with self.subTest(shape=shape, dtype=dtype):
                    self.assertEqual(len(chunk), len(shape))
                    # Never zero; never larger than the dataset on axes that
                    # actually have a size. A zero-sample layer still needs a
                    # chunk extent of at least 1 -- HDF5 forbids a zero dim.
                    for c, s in zip(chunk, shape):
                        self.assertGreaterEqual(c, 1)
                        if s > 0:
                            self.assertLessEqual(c, s)

    def test_chunk_stays_within_budget_when_the_budget_is_reachable(self):
        # The budget is an upper bound only when some chunk can meet it. The
        # current policy keeps the trailing axes whole, so a shape whose
        # trailing axes alone exceed the budget cannot: assert only where the
        # smallest chunk that policy can produce does fit.
        dtype = np.dtype(np.float32)
        for shape in self.shapes:
            chunk = choose_chunk_shape(shape, dtype)
            smallest = _nbytes((1, 1, *shape[2:]), dtype)
            with self.subTest(shape=shape):
                if smallest <= DEFAULT_TARGET_CHUNK_BYTES:
                    self.assertLessEqual(
                        _nbytes(chunk, dtype), DEFAULT_TARGET_CHUNK_BYTES
                    )

    def test_single_sample(self):
        chunk = choose_chunk_shape((1, 4096, 7, 7), np.dtype(np.float32))
        self.assertEqual(chunk[0], 1)

    def test_target_bytes_is_respected(self):
        dtype = np.dtype(np.float32)
        shape = (1200, 256, 13, 13)
        small = choose_chunk_shape(shape, dtype, target_bytes=64 * 1024)
        large = choose_chunk_shape(shape, dtype, target_bytes=4 * 1024 * 1024)
        self.assertLessEqual(_nbytes(small, dtype), 64 * 1024)
        self.assertLessEqual(_nbytes(large, dtype), 4 * 1024 * 1024)
        self.assertLess(_nbytes(small, dtype), _nbytes(large, dtype))

    def test_unknown_sample_count_ignores_the_current_size(self):
        # The contract of n_samples_known=False is exactly that the current
        # sample-axis size is not consulted, so two calls that differ only in
        # that size must agree. Stated this way the test does not depend on
        # which extent the policy happens to pick.
        dtype = np.dtype(np.float32)
        empty = choose_chunk_shape((0, 256, 13, 13), dtype, n_samples_known=False)
        huge = choose_chunk_shape(
            (10 ** 6, 256, 13, 13), dtype, n_samples_known=False
        )
        self.assertEqual(empty, huge)

    def test_both_sliceable_axes_are_chunked(self):
        # The promise of issue #144: reading a slice of channels must not have
        # to read every channel, which requires the feature axis to be split as
        # well as the sample axis. Asserted on a representative layer, since a
        # small one legitimately fits in a single chunk.
        shape = (1200, 256, 13, 13)
        chunk = choose_chunk_shape(shape, np.dtype(np.float32))
        self.assertLess(chunk[0], shape[0])
        self.assertLess(chunk[1], shape[1])

    def test_deterministic(self):
        shape = (1200, 256, 13, 13)
        dtype = np.dtype(np.float32)
        self.assertEqual(
            choose_chunk_shape(shape, dtype), choose_chunk_shape(shape, dtype)
        )

    def test_zero_sample_layer_still_gets_a_usable_chunk(self):
        # HDF5 forbids a zero chunk dimension, so an empty layer needs >= 1.
        chunk = choose_chunk_shape((0, 8), np.dtype(np.float32))
        self.assertEqual(chunk[0], 1)
        self.assertGreaterEqual(chunk[1], 1)

    def test_rejects_empty_feature_axis(self):
        for bad in ((5, 0), (5, 3, 0)):
            with self.subTest(shape=bad):
                with self.assertRaises(ValueError):
                    choose_chunk_shape(bad, np.dtype(np.float32))

    def test_rejects_negative_sample_axis(self):
        with self.assertRaises(ValueError):
            choose_chunk_shape((-1, 4), np.dtype(np.float32))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            choose_chunk_shape((100,), np.dtype(np.float32))
        with self.assertRaises(ValueError):
            choose_chunk_shape((100, 10), np.dtype(np.float32), target_bytes=0)


if __name__ == "__main__":
    unittest.main()
