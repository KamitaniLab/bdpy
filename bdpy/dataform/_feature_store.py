"""Storage backends for DNN feature collections.

``Features`` used to be welded to one on-disk layout: a directory per layer
holding one ``.mat`` file per stimulus. That layout has a file boundary only on
the sample axis, so reading a few channels still costs a full read of every
stimulus. This module puts a backend interface between ``Features`` and the
bytes, so the legacy layout and the chunked HDF5 layout are interchangeable and
a future backend needs no change to ``Features`` itself.

Every backend answers the same question -- "give me these labels, sliced this
way along the feature axes" -- and differs only in how much it has to read to do
it. The legacy backend loads whole stimulus files and slices in memory; the HDF5
backend pushes the slice down to h5py and reads only the chunks it covers.

This file is a part of BdPy.
"""

import glob
import os
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from . import _mat_v73
from ._feature_chunking import DEFAULT_TARGET_CHUNK_BYTES

__all__ = [
    "FeatureStore",
    "HDF5FeatureStore",
    "MatFeatureStore",
    "detect_format",
]

#: Root attribute marking a file as bdpy chunked feature storage.
FORMAT_ATTR = "bdpy_format"
#: Root attribute holding the schema version.
FORMAT_VERSION_ATTR = "bdpy_format_version"
#: Value of ``FORMAT_ATTR`` for feature files.
FORMAT_NAME = "features"
#: Highest schema version this module can read.
SUPPORTED_FORMAT_VERSION = 1
#: Dataset holding the feature array, shape ``(n_samples, *feature_shape)``.
FEATURES_DATASET = "features"
#: Dataset holding the stimulus labels, shape ``(n_samples,)``.
LABELS_DATASET = "labels"
#: Extension of a chunked feature file.
HDF5_EXT = "h5"

# A slice spec for the feature axes. Deliberately narrow: basic indexing only,
# i.e. what ``numpy.s_[128:256]`` or ``numpy.s_[8:16, 1:4]`` produces. See
# _validate_feature_slice for what is rejected and why.
# `types.EllipsisType` is 3.10+, and a bare "ellipsis" forward reference does not
# resolve at runtime (it is not a Python name), which makes
# typing.get_type_hints() raise. typeshed does define builtins.ellipsis, so the
# checker gets the real type and the interpreter gets an equivalent object.
if TYPE_CHECKING:
    from builtins import ellipsis
else:
    ellipsis = type(Ellipsis)
BasicIndexer = Union[slice, int, np.integer, ellipsis]
FeatureSlice = Union[BasicIndexer, Tuple[BasicIndexer, ...], None]


def _load_array_with_key(key: str, path: str) -> np.ndarray:
    """Load one array from a ``.mat`` file. Shared with :mod:`bdpy.dataform.features`.

    v5 ``.mat`` via scipy, v7.3 (HDF5) via h5py; avoids hdf5storage on the load
    path, which breaks under NumPy 2.0 (see :mod:`bdpy.dataform._mat_v73`).
    `key` comes first so ``partial(..., 'feat')`` is a ``Pool.map`` callable.
    """
    return _mat_v73.loadmat_key(path, key)


def _determine_num_parallel(num_files: int) -> int:
    # NOTE: optimal number of parallel processes is not clear. It could depend
    # on several factors such as the number of files, the size of files, the
    # number of cores, etc. For now, we use a simple heuristic based on the
    # number of files.
    num_parallel: int
    if num_files < 16:
        num_parallel = 1
    elif num_files < 64:
        num_parallel = 16
    else:
        num_parallel = 64
    return num_parallel


def _normalize_feature_slice(feature_slice: FeatureSlice) -> Tuple:
    """Normalize and validate a feature-axis slice spec.

    Returns a tuple of per-axis indexers; ``None`` means "no slice" and gives
    the empty tuple. Note this makes ``numpy.newaxis`` (which *is* ``None``)
    inexpressible, so a bare newaxis reads as "no slice"; inside a tuple it is
    rejected explicitly.
    """
    if feature_slice is None:
        return ()
    indexers = feature_slice if isinstance(feature_slice, tuple) else (feature_slice,)
    _validate_feature_slice(indexers)
    return indexers


def _validate_feature_slice(indexers: Tuple) -> None:
    """Reject anything outside basic, forward indexing.

    `feature_slice` exists to read part of a layer, not to reimplement NumPy
    indexing. Keeping it to basic forward indexing is what lets every backend
    mean the same thing by it: h5py handles slices with a positive step
    (negative start/stop included), integers and one ``Ellipsis`` exactly as
    NumPy does, but rejects a negative step outright. Allowing more would make
    the legacy and HDF5 backends diverge, or require reimplementing NumPy's
    semantics to keep them together.

    Parameters
    ----------
    indexers : tuple
        Per-axis indexers, as returned by :func:`_normalize_feature_slice`.

    Raises
    ------
    ValueError
        If any indexer is outside the supported set.
    """
    advice = (
        " feature_slice supports basic forward indexing only (slices with a "
        "positive step, integers and a single Ellipsis). Read without "
        "feature_slice and index the resulting array with NumPy instead."
    )
    if sum(ix is Ellipsis for ix in indexers) > 1:
        raise ValueError("feature_slice may contain at most one Ellipsis." + advice)
    for ix in indexers:
        # bool is a subclass of int in Python, so it must be rejected *before*
        # the integer check or True would silently be read as index 1.
        if isinstance(ix, (bool, np.bool_)):
            raise ValueError(
                "feature_slice does not support boolean indexing." + advice
            )
        if ix is Ellipsis or isinstance(ix, (int, np.integer)):
            continue
        if isinstance(ix, slice):
            for part, value in (
                ("start", ix.start), ("stop", ix.stop), ("step", ix.step),
            ):
                if value is None:
                    continue
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)
                ):
                    raise ValueError(
                        "feature_slice slice {} must be an integer or None, "
                        "got {!r}.".format(part, value) + advice
                    )
            if ix.step is not None and ix.step < 1:
                raise ValueError(
                    "feature_slice does not support a step below 1 "
                    "(got {}).".format(ix.step) + advice
                )
            continue
        if ix is None:
            raise ValueError(
                "feature_slice does not support numpy.newaxis." + advice
            )
        raise ValueError(
            "feature_slice does not support {} indexing.".format(
                type(ix).__name__
            )
            + advice
        )


class FeatureStore(ABC):
    """Interface to a collection of DNN features on disk.

    A store exposes layers and stimulus labels, and reads features for a set of
    labels with an optional slice along the feature axes. Implementations differ
    only in how much I/O a given request costs.
    """

    @property
    @abstractmethod
    def layers(self) -> List[str]:
        """List of DNN layers held by this store."""

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """List of stimulus labels, in the store's own order."""

    @abstractmethod
    def shape(self, layer: str) -> Tuple[int, ...]:
        """Full shape ``(n_samples, *feature_shape)`` of `layer`."""

    @abstractmethod
    def dtype(self, layer: str) -> np.dtype:
        """Dtype of the features in `layer`."""

    @abstractmethod
    def read(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        feature_slice: FeatureSlice = None,
    ) -> np.ndarray:
        """Read features from `layer`.

        Parameters
        ----------
        layer : str
            DNN layer.
        labels : sequence of str, optional
            Stimulus labels to read. ``None`` reads every label in store order.
            Otherwise rows come back **in the order given**, and repeated labels
            yield repeated rows.
        feature_slice : slice, int, Ellipsis or tuple of those, optional
            Index applied to the feature axes (axes 1 and up), as produced by
            ``numpy.s_[...]``. Basic forward indexing only: slices with a
            positive step and integer bounds, integers, and at most one
            ``Ellipsis``. Fancy indexing, a negative step, a non-integer slice
            bound, booleans, and ``numpy.newaxis`` *inside a tuple* raise
            ``ValueError``. A bare ``numpy.newaxis`` is ``None``, the "no
            slice" default, so it reads the whole feature tensor.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_labels, *sliced_feature_shape)``.
        """

    def chunk_extent(self, layer: str, axis: int) -> Optional[int]:
        """On-disk chunk extent of `layer` along `axis`, if the store has one.

        Returns ``None`` when the backend has no chunking to align with, in
        which case :meth:`iter_chunks` falls back to a byte budget.
        """
        return None

    def iter_chunks(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        axis: int = 1,
        size: Optional[int] = None,
    ) -> Iterator[Tuple[slice, np.ndarray]]:
        """Iterate over `layer` in slabs along `axis`.

        Yields ``(sl, block)`` pairs, where `sl` is the slice applied to `axis`
        of the selected array and `block` is that slab. The slice is yielded so
        that callers can place results back without tracking offsets themselves.

        There is deliberately no `feature_slice` here. Composing an arbitrary
        slice with the per-slab slice means reimplementing NumPy's index
        arithmetic, which is both the source of subtle wrong results and more
        generality than reading a layer in slabs needs. Slice the blocks as they
        come out instead.

        Parameters
        ----------
        layer : str
            DNN layer.
        labels : sequence of str, optional
            Stimulus labels, as in :meth:`read`.
        axis : int, optional
            Axis to iterate over. Axis 0 is the sample axis; the default, axis
            1, is the outermost feature axis, which is the one partial feature
            reads are about.
        size : int, optional
            Number of elements per slab. Defaults to the on-disk chunk extent
            along `axis` when the backend has one, so that iteration is
            chunk-aligned and every element is read exactly once; otherwise a
            size derived from the same byte budget is used.

        Yields
        ------
        tuple of (slice, numpy.ndarray)
            The slice applied to `axis`, and the corresponding slab.
        """
        selected_shape = self._selected_shape(layer, labels)
        if axis < 0:
            axis += len(selected_shape)
        if not 0 <= axis < len(selected_shape):
            raise ValueError(
                "axis {} is out of range for selected shape {}".format(
                    axis, selected_shape
                )
            )

        length = selected_shape[axis]
        if size is None:
            size = self._default_iter_size(layer, axis, selected_shape)
        if size < 1:
            raise ValueError("size must be positive, got {}".format(size))

        for start in range(0, length, size):
            sl = slice(start, min(start + size, length))
            if axis == 0:
                block_labels = (
                    list(self.labels) if labels is None else list(labels)
                )[sl]
                yield sl, self.read(layer, block_labels)
            else:
                # axis >= 1 indexes the feature axes, where axis 1 of the array
                # is entry 0 of the feature-slice tuple.
                indexers = (slice(None),) * (axis - 1) + (sl,)
                yield sl, self.read(layer, labels, indexers)

    def _selected_shape(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[int, ...]:
        """Shape the selection would have, without reading the data."""
        full = self.shape(layer)
        n_samples = full[0] if labels is None else len(labels)
        return (n_samples, *full[1:])

    def _default_iter_size(
        self, layer: str, axis: int, selected_shape: Tuple[int, ...]
    ) -> int:
        extent = self.chunk_extent(layer, axis)
        if extent is not None:
            return max(1, min(extent, selected_shape[axis]))
        # No on-disk chunking to align with: fall back to the byte budget.
        itemsize = np.dtype(self.dtype(layer)).itemsize
        other = 1
        for i, n in enumerate(selected_shape):
            if i != axis:
                other *= int(n)
        per_element = max(1, other * itemsize)
        return max(1, min(selected_shape[axis], DEFAULT_TARGET_CHUNK_BYTES // per_element))


class MatFeatureStore(FeatureStore):
    """Legacy per-stimulus ``.mat`` feature directory.

    Layout is ``<dpath>/<layer>/<label>.<ext>``, each file holding one sample
    under `key` with a leading sample axis. There is no file boundary on the
    feature axes, so `feature_slice` is applied after the full stimulus files
    have been loaded and concatenated -- the read cost is the same as before,
    and the slice only saves the caller from doing it themselves.

    Parameters
    ----------
    dpath : str
        Feature directory.
    ext : str, optional
        Feature file extension (default: ``'mat'``).
    key : str, optional
        Variable name inside each file (default: ``'feat'``).
    """

    def __init__(self, dpath: str, ext: str = "mat", key: str = "feat"):
        self._dpath = dpath
        self._ext = ext
        self._key = key
        self._layers = self._collect_layers()
        self._labels = self._collect_labels()
        self._file_table: Dict[str, Dict[str, str]] = {
            layer: {
                label: os.path.join(dpath, layer, label + "." + ext)
                for label in self._labels
            }
            for layer in self._layers
        }

    @property
    def layers(self) -> List[str]:
        return self._layers

    @property
    def labels(self) -> List[str]:
        return self._labels

    def path(self, layer: str, label: str) -> str:
        """Path of the file holding `label` in `layer`."""
        return self._file_table[layer][label]

    def shape(self, layer: str) -> Tuple[int, ...]:
        if not self._labels:
            raise RuntimeError("No features found in {}".format(self._dpath))
        sample = _load_array_with_key(self._key, self.path(layer, self._labels[0]))
        return (len(self._labels), *sample.shape[1:])

    def dtype(self, layer: str) -> np.dtype:
        if not self._labels:
            raise RuntimeError("No features found in {}".format(self._dpath))
        sample = _load_array_with_key(self._key, self.path(layer, self._labels[0]))
        return sample.dtype

    def read(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        feature_slice: FeatureSlice = None,
    ) -> np.ndarray:
        if labels is None:
            labels = self._labels
        paths = [self.path(layer, label) for label in labels]

        num_parallel = _determine_num_parallel(len(paths))
        load = partial(_load_array_with_key, self._key)
        if num_parallel == 1:
            arrays = list(map(load, paths))
        else:
            with Pool(processes=num_parallel) as pool:
                arrays = pool.map(load, paths)
        features = np.concatenate(arrays, axis=0)

        indexers = _normalize_feature_slice(feature_slice)
        if indexers:
            features = features[(slice(None), *indexers)]
        return features

    def _collect_layers(self) -> List[str]:
        return sorted(
            d
            for d in os.listdir(self._dpath)
            if os.path.isdir(os.path.join(self._dpath, d))
        )

    def _collect_labels(self) -> List[str]:
        labels: List[str] = []
        for layer in self._layers:
            layer_dir = os.path.join(self._dpath, layer)
            layer_dir = layer_dir.replace("[", "[[]")  # Use glob.escape for Python 3.4 or later
            files = glob.glob(os.path.join(layer_dir, "*." + self._ext))
            labels_t = sorted(
                os.path.splitext(os.path.basename(f))[0] for f in files
            )
            if not labels:
                labels = labels_t
            elif labels != labels_t:
                raise RuntimeError("Invalid feature file in %s " % self._dpath)
        return labels


class HDF5FeatureStore(FeatureStore):
    """Chunked HDF5 feature directory, one ``<layer>.h5`` file per layer.

    Each file holds ``/features`` with shape ``(n_samples, *feature_shape)`` and
    ``/labels`` with the matching stimulus labels. ``/features`` is explicitly
    chunked, so a slice along the feature axes reads only the chunks it covers
    instead of the whole layer.

    Parameters
    ----------
    dpath : str
        Directory holding ``<layer>.h5`` files.
    """

    def __init__(self, dpath: str):
        self._dpath = dpath
        self._layers = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(glob.escape(dpath), "*." + HDF5_EXT))
        )
        if not self._layers:
            raise RuntimeError("No .{} feature file found in {}".format(HDF5_EXT, dpath))
        self._labels: List[str] = []
        self._label_index: Dict[str, int] = {}
        self._validate_all()

    @property
    def layers(self) -> List[str]:
        return self._layers

    @property
    def labels(self) -> List[str]:
        return self._labels

    def path(self, layer: str) -> str:
        """Path of the file holding `layer`."""
        return os.path.join(self._dpath, layer + "." + HDF5_EXT)

    def shape(self, layer: str) -> Tuple[int, ...]:
        with self._open(layer) as f:
            return tuple(int(n) for n in f[FEATURES_DATASET].shape)

    def dtype(self, layer: str) -> np.dtype:
        with self._open(layer) as f:
            dtype: np.dtype = np.dtype(f[FEATURES_DATASET].dtype)
        return dtype

    def chunk_extent(self, layer: str, axis: int) -> Optional[int]:
        with self._open(layer) as f:
            chunks = f[FEATURES_DATASET].chunks
        if chunks is None or axis >= len(chunks):
            return None
        return int(chunks[axis])

    def read(
        self,
        layer: str,
        labels: Optional[Sequence[str]] = None,
        feature_slice: FeatureSlice = None,
    ) -> np.ndarray:
        indexers = _normalize_feature_slice(feature_slice)

        with self._open(layer) as f:
            dset = f[FEATURES_DATASET]

            if labels is None:
                return self._read_rows(dset, slice(None), indexers)

            rows = self._row_indices(labels)
            if rows.size == dset.shape[0] and np.array_equal(
                rows, np.arange(dset.shape[0])
            ):
                # Asking for every row in order: a plain slice reads
                # contiguously, where a full index list would not.
                return self._read_rows(dset, slice(None), indexers)
            # h5py wants a strictly increasing index list and allows at most one
            # fancy index per selection. Read each distinct row once in order,
            # then restore the caller's order (and any repeats) with `inverse`.
            uniq, inverse = np.unique(rows, return_inverse=True)
            block = self._read_rows(dset, [int(i) for i in uniq], indexers)
            if uniq.size == rows.size and np.array_equal(uniq, rows):
                return block
            return block[inverse]

    @staticmethod
    def _read_rows(
        dset: h5py.Dataset, rows: Union[slice, List[int]], indexers: Tuple
    ) -> np.ndarray:
        """Read `rows` from `dset`, pushing `indexers` down into the h5py selection."""
        if not indexers:
            return np.asarray(dset[rows])
        # Basic forward indexers are not fancy, so they ride along in the same
        # selection as the row list and h5py reads only the chunks they cover.
        # _validate_feature_slice has already rejected anything else, which is
        # what keeps this to a single line.
        return np.asarray(dset[(rows, *indexers)])

    def _row_indices(self, labels: Sequence[str]) -> np.ndarray:
        try:
            return np.array([self._label_index[label] for label in labels], dtype=int)
        except KeyError as exc:
            raise KeyError(
                "Label {} not found in {}".format(exc.args[0], self._dpath)
            ) from None

    def _open(self, layer: str) -> h5py.File:
        return h5py.File(self.path(layer), "r")

    def _validate_all(self) -> None:
        """Validate every layer file and pin down the shared label sequence.

        A row index built from one layer is used against every layer's
        ``/features``, so the layers must agree on the labels *and on their
        order*; otherwise a label lookup would silently return another
        stimulus' row. The legacy ``.mat`` backend enforces the same invariant
        in :meth:`MatFeatureStore._collect_labels`.
        """
        reference: Optional[List[str]] = None
        reference_layer = ""
        for layer in self._layers:
            with self._open(layer) as f:
                labels = _validate_format(f, self.path(layer))
            if reference is None:
                reference, reference_layer = labels, layer
            elif labels != reference:
                raise RuntimeError(
                    _label_mismatch_message(
                        self._dpath, reference_layer, reference, layer, labels
                    )
                )

        assert reference is not None  # self._layers is non-empty
        self._labels = reference
        self._label_index = {label: i for i, label in enumerate(reference)}


def _validate_format(f: h5py.File, path: str) -> List[str]:
    """Check that `f` is well-formed bdpy feature storage, and return its labels.

    Everything a reader relies on is checked in one place: the format marker,
    a version in the range this build understands, both datasets present with
    the expected rank, one label per feature row, and no duplicate labels.

    Duplicate labels matter more than they look. Labels are mapped to row
    indices once, so a repeated label would silently resolve every occurrence
    to the last row and drop the earlier one. The legacy ``.mat`` layout cannot
    express duplicates at all -- the file name *is* the label -- so rejecting
    them keeps the two layouts equivalent.

    Parameters
    ----------
    f : h5py.File
        Open feature file.
    path : str
        Its path, for error messages.

    Returns
    -------
    list of str
        The file's labels, already decoded.

    Raises
    ------
    RuntimeError
        If any of the above does not hold.
    """
    fmt = f.attrs.get(FORMAT_ATTR)
    if isinstance(fmt, bytes):
        fmt = fmt.decode("utf-8")
    if fmt != FORMAT_NAME:
        raise RuntimeError(
            "{} is not bdpy feature storage ({} = {!r}); expected {!r}".format(
                path, FORMAT_ATTR, fmt, FORMAT_NAME
            )
        )

    if FORMAT_VERSION_ATTR not in f.attrs:
        raise RuntimeError(
            "{} has no {} attribute, so it is not valid bdpy feature "
            "storage.".format(path, FORMAT_VERSION_ATTR)
        )
    raw_version = f.attrs[FORMAT_VERSION_ATTR]
    # Require an actual integer scalar rather than coercing: int(1.5) would
    # silently round a malformed version down to a supported one. bool is
    # checked first because it is a subclass of int.
    if isinstance(raw_version, (bool, np.bool_)) or not isinstance(
        raw_version, (int, np.integer)
    ):
        raise RuntimeError(
            "{} has a malformed {} ({!r}); expected an integer.".format(
                path, FORMAT_VERSION_ATTR, raw_version
            )
        )
    version = int(raw_version)
    if version < 1:
        raise RuntimeError(
            "{} declares feature storage format version {}; versions start at "
            "1.".format(path, version)
        )
    if version > SUPPORTED_FORMAT_VERSION:
        raise RuntimeError(
            "{} uses feature storage format version {}, but this version of "
            "bdpy supports up to version {}. Please upgrade bdpy.".format(
                path, version, SUPPORTED_FORMAT_VERSION
            )
        )

    for name in (FEATURES_DATASET, LABELS_DATASET):
        if name not in f:
            raise RuntimeError("{} has no /{} dataset".format(path, name))

    features, labels_dset = f[FEATURES_DATASET], f[LABELS_DATASET]
    if features.ndim < 2:
        raise RuntimeError(
            "{}: /{} must have a sample axis and at least one feature axis, "
            "got shape {}".format(path, FEATURES_DATASET, features.shape)
        )
    if labels_dset.ndim != 1:
        raise RuntimeError(
            "{}: /{} must be one-dimensional, got shape {}".format(
                path, LABELS_DATASET, labels_dset.shape
            )
        )

    labels = _decode_labels(labels_dset[()])
    if len(labels) != features.shape[0]:
        raise RuntimeError(
            "{}: /{} has {} rows but /{} has {} entries".format(
                path, FEATURES_DATASET, features.shape[0],
                LABELS_DATASET, len(labels),
            )
        )
    duplicates = _duplicates(labels)
    if duplicates:
        raise RuntimeError(
            "{}: /{} contains duplicate labels ({}). Labels identify rows, so "
            "they must be unique.".format(
                path, LABELS_DATASET, ", ".join(repr(d) for d in duplicates[:3])
            )
        )
    return labels


def _duplicates(labels: Sequence[str]) -> List[str]:
    """Labels appearing more than once, in first-seen order."""
    seen: Dict[str, int] = {}
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
    return [label for label, n in seen.items() if n > 1]


def _label_mismatch_message(
    dpath: str,
    reference_layer: str,
    reference: List[str],
    layer: str,
    labels: List[str],
) -> str:
    """Describe *how* two layers' label sequences differ, not just that they do."""
    if len(labels) != len(reference):
        detail = "{} has {} labels, {} has {}".format(
            reference_layer, len(reference), layer, len(labels)
        )
    elif sorted(labels) == sorted(reference):
        first = next(
            i for i, (a, b) in enumerate(zip(reference, labels)) if a != b
        )
        detail = (
            "same labels in a different order; first difference at index {}: "
            "{} has {!r}, {} has {!r}".format(
                first, reference_layer, reference[first], layer, labels[first]
            )
        )
    else:
        only_ref = sorted(set(reference) - set(labels))[:3]
        only_this = sorted(set(labels) - set(reference))[:3]
        detail = "different labels; only in {}: {}, only in {}: {}".format(
            reference_layer, only_ref or "-", layer, only_this or "-"
        )
    return (
        "Inconsistent labels across layers in {}: {}. Every layer must hold the "
        "same stimulus labels in the same order.".format(dpath, detail)
    )


def _decode_labels(raw: np.ndarray) -> List[str]:
    """Decode an HDF5 string dataset into a list of str (h5py 3.x gives bytes)."""
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(raw).ravel().tolist()
    ]


def detect_format(dpath: str, ext: str = "mat") -> str:
    """Work out which storage layout `dpath` holds.

    The two layouts are told apart by the files that are actually there, not by
    the mere presence of a subdirectory: a legacy tree is one whose
    subdirectories really contain ``*.<ext>`` feature files, so an unrelated
    subdirectory next to ``<layer>.h5`` files does not turn a chunked directory
    into a legacy one.

    A directory holding both is ambiguous and raises rather than picking one
    silently -- guessing wrong means reading different data than the caller
    meant. Pass an explicit `format` to resolve it.

    Parameters
    ----------
    dpath : str
        Feature directory.
    ext : str, optional
        Extension of the legacy per-stimulus files (default: ``'mat'``).

    Returns
    -------
    str
        ``'hdf5'`` or ``'mat'``.

    Raises
    ------
    RuntimeError
        If the directory holds both layouts, or neither.
    """
    escaped = glob.escape(dpath)
    has_h5 = bool(glob.glob(os.path.join(escaped, "*." + HDF5_EXT)))
    has_legacy = any(
        glob.glob(os.path.join(glob.escape(os.path.join(dpath, d)), "*." + ext))
        for d in os.listdir(dpath)
        if os.path.isdir(os.path.join(dpath, d))
    )

    if has_h5 and has_legacy:
        raise RuntimeError(
            "{} holds both chunked .{} files and a legacy .{} tree, so the "
            "storage format is ambiguous. Pass format='hdf5' or format='mat' "
            "to choose one.".format(dpath, HDF5_EXT, ext)
        )
    if has_h5:
        return "hdf5"
    if has_legacy:
        return "mat"
    raise RuntimeError(
        "No features found in {}: expected either <layer>.{} files or "
        "<layer>/<label>.{} subdirectories.".format(dpath, HDF5_EXT, ext)
    )
