"""Chunk-shape policy for chunked feature storage.

HDF5 reads whole chunks even when only a few elements of a chunk are selected,
so the chunk shape - not the file format alone - is what decides the cost of a
partial read. This module turns an array shape and dtype into a chunk shape
under a byte budget, rather than hard-coding an extent along any one axis.

This file is a part of BdPy.
"""

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = ["DEFAULT_TARGET_CHUNK_BYTES", "choose_chunk_shape"]

#: Default per-chunk byte budget. h5py's own guidance puts a useful chunk
#: somewhere between 10 KiB and 1 MiB; features are large and read in slabs, so
#: we sit at the top of that range.
DEFAULT_TARGET_CHUNK_BYTES = 1 << 20


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def choose_chunk_shape(
    shape: Sequence[int],
    dtype: np.dtype,
    target_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
    n_samples_known: bool = True,
) -> Tuple[int, ...]:
    """Choose an HDF5 chunk shape for a feature array under a byte budget.

    The two axes callers actually slice are axis 0 (samples) and axis 1 (the
    outermost feature axis, typically channels or units), so both are chunked.
    Trailing axes - the spatial axes of a convolutional feature map, say
    ``(13, 13)`` - are kept whole: they are small and are never sliced in
    practice, and splitting them would only shrink the chunks.

    Within the budget the two chunked axes are split roughly evenly in element
    count, so that neither a sample-wise read nor a channel-wise read degenerates
    into reading the entire dataset. Each extent is then shrunk to the smallest
    one needing the same number of chunks, which keeps HDF5 from padding the
    edge chunks and inflating the file.

    Parameters
    ----------
    shape : sequence of int
        Full array shape, ``(n_samples, *feature_shape)``.
    dtype : numpy.dtype
        Array dtype; only its itemsize matters.
    target_bytes : int, optional
        Size budget for one chunk, in bytes
        (default: ``DEFAULT_TARGET_CHUNK_BYTES``). It is an upper bound only
        when the budget can be met at all: the trailing axes are kept whole, so
        a shape whose trailing axes alone exceed `target_bytes` necessarily
        produces a larger chunk.
    n_samples_known : bool, optional
        Whether ``shape[0]`` is the final number of samples. Pass ``False`` when
        writing to a resizable dataset whose sample count is not yet known, so
        that the sample-axis extent is not capped by the current size.

    Returns
    -------
    tuple of int
        Chunk shape, same length as ``shape``. Every element is at least 1, and
        no larger than the corresponding entry of ``shape`` for axes of
        non-zero size (a layer with zero samples still needs a chunk extent of
        at least 1 on the sample axis, since HDF5 forbids a zero chunk dim).

    Raises
    ------
    ValueError
        If `shape` has fewer than two axes, any feature axis has size 0, the
        sample axis is negative, or `target_bytes` is not positive.

    Examples
    --------
    >>> choose_chunk_shape((1200, 1000), np.dtype(np.float32))
    (400, 500)
    >>> choose_chunk_shape((1200, 256, 13, 13), np.dtype(np.float32))
    (39, 37, 13, 13)
    >>> choose_chunk_shape((50, 1000), np.dtype(np.float32))
    (50, 1000)
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        raise ValueError(
            "Feature arrays need at least a sample axis and a feature axis; "
            "got shape {}".format(shape)
        )
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive, got {}".format(target_bytes))
    if shape[0] < 0:
        raise ValueError("sample axis cannot be negative, got {}".format(shape))
    if any(s < 1 for s in shape[1:]):
        # A zero-width feature axis carries nothing and cannot be chunked.
        # A zero-sample layer, by contrast, is meaningful and is allowed below.
        raise ValueError(
            "feature axes must all be non-empty, got shape {}".format(shape)
        )

    itemsize = np.dtype(dtype).itemsize

    # The whole array fits in one chunk: nothing to gain from splitting it.
    # Clamp to at least 1 per axis -- a zero-sample layer is legal but HDF5
    # rejects a chunk with a zero dimension.
    if n_samples_known and _prod(shape) * itemsize <= target_bytes:
        return tuple(max(1, s) for s in shape)

    # Bytes taken by one (sample, feature-0) cell, i.e. one element of the two
    # chunked axes with the trailing axes kept whole.
    inner = max(1, _prod(shape[2:])) * itemsize
    budget_cells = max(1, target_bytes // inner)

    # Split the budget evenly between the two chunked axes, then let each axis
    # absorb the headroom the other did not need: a short sample axis should not
    # force a narrow feature chunk, and vice versa.
    c1 = min(shape[1], max(1, math.isqrt(budget_cells)))
    c0 = max(1, budget_cells // c1)
    if n_samples_known and shape[0] > 0:
        c0 = min(shape[0], c0)
    c1 = min(shape[1], max(c1, budget_cells // c0))

    # HDF5 allocates whole chunks, so a chunk extent that divides its axis
    # unevenly pads the edge chunks and inflates the file. Shrinking the extent
    # to the smallest one that needs the same number of chunks removes most of
    # that padding, and can only lower the chunk size, so the budget still holds.
    c1 = _snap(shape[1], c1)
    if n_samples_known and shape[0] > 0:
        c0 = _snap(shape[0], c0)

    return (c0, c1, *shape[2:])


def _snap(length: int, chunk: int) -> int:
    """Smallest extent covering `length` in the same number of chunks as `chunk`."""
    if length < 1 or chunk >= length:
        return chunk
    n_chunks = -(-length // chunk)  # ceil
    return -(-length // n_chunks)   # ceil
