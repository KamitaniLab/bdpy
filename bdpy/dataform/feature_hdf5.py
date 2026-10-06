"""Chunked HDF5 storage for DNN features.

The legacy feature layout writes one ``.mat`` file per stimulus, so the only
file boundary is the sample axis and a partial read along the feature axes is
impossible. This module writes the chunked layout instead: one ``<layer>.h5``
per layer, holding the whole layer in a dataset that is explicitly chunked on
both the sample axis and the outermost feature axis.

Schema version 1::

    <layer>.h5
        /features   (n_samples, *feature_shape)
        /labels     (n_samples,)   variable-length UTF-8
        attrs:
            bdpy_format         = "features"
            bdpy_format_version = 1

One file per layer rather than one file for everything: layers differ in shape
and in optimal chunk shape, and separate files keep regeneration, copying and
parallel writing per-layer.

Compression is off by default. The point of this format is partial-read latency,
and every compressed chunk costs a decompression on the way out; pass
``compression=`` when size matters more than speed.

This file is a part of BdPy.

Examples
--------
Write a whole layer at once::

    save_features('features/conv5.h5', array, labels)

Write incrementally, as feature extraction produces one stimulus at a time::

    with FeatureWriter('features/conv5.h5', feature_shape=(256, 13, 13),
                       dtype=np.float32) as writer:
        for label, feature in extract():
            writer.append(feature, label)

Migrate an existing ``.mat`` tree::

    convert_features_to_hdf5('features_mat', 'features_h5')
"""

import errno
import os
import uuid
import weakref
from types import TracebackType
from typing import Iterable, Optional, Sequence, Set, Tuple, Type

import h5py
import numpy as np

from ._feature_chunking import DEFAULT_TARGET_CHUNK_BYTES, choose_chunk_shape
from ._feature_store import (
    FEATURES_DATASET,
    FORMAT_ATTR,
    FORMAT_NAME,
    FORMAT_VERSION_ATTR,
    HDF5_EXT,
    LABELS_DATASET,
    SUPPORTED_FORMAT_VERSION,
    MatFeatureStore,
    _duplicates,
)

__all__ = [
    "FeatureWriter",
    "convert_features_to_hdf5",
    "save_features",
]

#: Chunk length of the resizable labels dataset.
_LABELS_CHUNK = 64


def _string_dtype() -> np.dtype:
    dtype: np.dtype = h5py.string_dtype(encoding="utf-8")
    return dtype


def _write_header(f: h5py.File) -> None:
    """Stamp the format marker and version. The layer name is the file name."""
    f.attrs[FORMAT_ATTR] = FORMAT_NAME
    f.attrs[FORMAT_VERSION_ATTR] = SUPPORTED_FORMAT_VERSION


#: Suffix of the scratch file a write is staged in.
_PARTIAL_SUFFIX = ".partial"
#: Cap on the copied basename so the staged name stays inside NAME_MAX.
_MAX_STEM = 64


def _staging_path(path: str) -> str:
    """Name a scratch file next to `path`, invisible to feature-directory scans.

    The name is dotted and does not end in ``.h5``, so neither
    ``glob('*.h5')`` in :class:`~bdpy.dataform._feature_store.HDF5FeatureStore`
    nor :func:`~bdpy.dataform._feature_store.detect_format` picks it up as a
    layer while a write is in flight. The random component lets two processes
    stage the same layer without colliding.
    """
    directory, name = os.path.split(os.path.abspath(path))
    return os.path.join(
        directory,
        ".{}.{}{}".format(name[:_MAX_STEM], uuid.uuid4().hex, _PARTIAL_SUFFIX),
    )


def _prepare_target(path: str, overwrite: bool) -> str:
    """Refuse to clobber, then name a scratch file next to `path`.

    Writing goes to a scratch file in the *same directory*, so publishing it
    (see :func:`_promote`) is a single atomic step on the same filesystem:
    either the finished file appears at `path`, or nothing does. A half-written
    file must never be left where a reader -- or the converter's skip-if-exists
    check -- would take it for a complete one.

    This check fails early, before any work is done; :func:`_promote` repeats
    it when publishing, since another writer may publish `path` meanwhile.

    The caller opens the returned path with mode ``"x"``, which both creates it
    exclusively and lets HDF5 apply the process umask, so published files keep
    the permissions they had before staging existed.
    """
    if os.path.exists(path) and not overwrite:
        raise _exists_error(path)
    _makedirs_for(path)
    return _staging_path(path)


def _exists_error(path: str) -> FileExistsError:
    return FileExistsError(
        "{} already exists. Pass overwrite=True to replace it.".format(path)
    )


#: errno values with which os.link reports that the filesystem cannot hard-link.
_NO_HARDLINK_ERRNOS = frozenset(
    {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
)


def _promote(tmp_path: str, path: str, overwrite: bool) -> None:
    """Publish the finished scratch file at `path`.

    With `overwrite`, an atomic rename replaces whatever is there. Without it,
    the file is published with :func:`os.link`, which fails if `path` exists:
    checking for a file and publishing ours are one atomic step, so a file that
    another writer published while this one was staging is never clobbered.

    On a filesystem that cannot hard-link, this falls back to checking for
    `path` and then renaming, which leaves a short window between the two.

    Raises
    ------
    FileExistsError
        If `path` exists and `overwrite` is False.
    """
    if overwrite:
        os.replace(tmp_path, path)
        return
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        raise _exists_error(path) from None
    except OSError as exc:
        if exc.errno not in _NO_HARDLINK_ERRNOS:
            raise
        if os.path.exists(path):
            raise _exists_error(path) from None
        os.replace(tmp_path, path)
    else:
        _discard(tmp_path)


def _discard(tmp_path: str) -> None:
    """Remove the scratch file, ignoring an already-vanished one."""
    try:
        os.remove(tmp_path)
    except OSError:
        pass


def save_features(
    path: str,
    features: np.ndarray,
    labels: Sequence[str],
    chunks: Optional[Tuple[int, ...]] = None,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
    compression: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    overwrite: bool = False,
) -> None:
    """Write a whole layer to a chunked HDF5 feature file.

    Parameters
    ----------
    path : str
        Output file. An existing file -- including one another writer
        publishes meanwhile -- is kept unless `overwrite` is True. The write is
        atomic: on failure nothing is left at `path`.
    features : numpy.ndarray
        Feature array of shape ``(n_samples, *feature_shape)``.
    labels : sequence of str
        One stimulus label per sample, in the same order as `features`.
    chunks : tuple of int, optional
        Explicit chunk shape. Defaults to :func:`choose_chunk_shape`.
    target_chunk_bytes : int, optional
        Per-chunk byte budget used when `chunks` is not given.
    compression : str, optional
        h5py compression filter, e.g. ``'lzf'`` or ``'gzip'``. ``None``
        (default) stores the features uncompressed, which keeps partial reads
        fast.
    dtype : numpy.dtype, optional
        Dtype to store. Defaults to the dtype of `features`.
    overwrite : bool, optional
        Replace an existing file at `path` (default: False).

    Raises
    ------
    ValueError
        If `labels` does not have one entry per sample, or `features` has fewer
        than two axes.
    FileExistsError
        If `path` exists or is published meanwhile, and `overwrite` is False.
    """
    features = np.asarray(features)
    if features.ndim < 2:
        raise ValueError(
            "features need a sample axis and at least one feature axis; "
            "got shape {}".format(features.shape)
        )
    labels = list(labels)
    if len(labels) != features.shape[0]:
        raise ValueError(
            "got {} labels for {} samples".format(len(labels), features.shape[0])
        )

    # One write path, not two: writing a whole layer is the degenerate case of
    # writing it incrementally, and keeping them separate is what let a bug
    # (a zero-sample layer) land in only one of them.
    with FeatureWriter(
        path,
        feature_shape=features.shape[1:],
        dtype=features.dtype if dtype is None else dtype,
        n_samples=features.shape[0],
        chunks=chunks,
        target_chunk_bytes=target_chunk_bytes,
        compression=compression,
        overwrite=overwrite,
    ) as writer:
        writer.extend(features, labels)


class FeatureWriter:
    """Incremental writer for a chunked HDF5 feature file.

    Feature extraction produces one stimulus at a time, so the datasets are
    created resizable (``maxshape=(None, *feature_shape)``) and grown in blocks
    as samples arrive.

    Writing goes to a temporary file next to `path`; :meth:`close` moves it into
    place and :meth:`abort` throws it away, so a failed write leaves nothing at
    `path`. **Prefer the context manager**, which aborts when the block raises --
    a bare ``try/finally: writer.close()`` would publish a half-written file.

    Parameters
    ----------
    path : str
        Output file. An existing file -- including one another writer
        publishes meanwhile -- is kept unless `overwrite` is True, and nothing
        is written there until :meth:`close` succeeds. On a filesystem without
        hard links, a short window remains in which a file published at the
        same moment can still be replaced.
    feature_shape : sequence of int
        Shape of a single sample's features, without the sample axis.
    dtype : numpy.dtype
        Dtype to store.
    n_samples : int, optional
        Expected number of samples, if known. Used only to pick a chunk shape.
    chunks : tuple of int, optional
        Explicit chunk shape, overriding the byte budget.
    target_chunk_bytes : int, optional
        Per-chunk byte budget used when `chunks` is not given.
    compression : str, optional
        h5py compression filter. ``None`` (default) stores uncompressed.
    overwrite : bool, optional
        Replace an existing file at `path` (default: False).

    Raises
    ------
    FileExistsError
        If `path` exists, or (from :meth:`close`) was published meanwhile,
        and `overwrite` is False.

    Examples
    --------
    >>> with FeatureWriter('conv5.h5', (256, 13, 13), np.float32) as w:  # doctest: +SKIP
    ...     w.append(feature, 'img0001')
    """

    def __init__(
        self,
        path: str,
        feature_shape: Sequence[int],
        dtype: np.dtype,
        n_samples: Optional[int] = None,
        chunks: Optional[Tuple[int, ...]] = None,
        target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
        compression: Optional[str] = None,
        overwrite: bool = False,
    ):
        self._feature_shape = tuple(int(s) for s in feature_shape)
        if not self._feature_shape:
            raise ValueError("feature_shape must have at least one axis")
        self._dtype = np.dtype(dtype)
        self._n = 0
        # Labels identify rows, so they must be unique across the whole file --
        # not just within one call. A repeated label would make every lookup
        # resolve to the last row and silently drop the earlier one.
        self._seen: Set[str] = set()

        if chunks is None:
            chunks = choose_chunk_shape(
                (n_samples if n_samples is not None else 0, *self._feature_shape),
                self._dtype,
                target_bytes=target_chunk_bytes,
                n_samples_known=n_samples is not None,
            )
        # Fail before doing any work if the target is occupied, then write to
        # a scratch file so `path` stays untouched until close() succeeds.
        self._path = path
        self._overwrite = overwrite
        self._aborted = False
        self._tmp_path: Optional[str] = _prepare_target(path, overwrite)
        self._file: Optional[h5py.File] = h5py.File(self._tmp_path, "x")
        # If the writer is dropped without close() or abort(), the scratch file
        # would otherwise linger next to the output. weakref.finalize (rather
        # than __del__) runs exactly once, is detachable at publish time, and
        # closes over the path instead of the writer, so it can never resurrect
        # the object or outlive a name it no longer owns.
        self._finalizer = weakref.finalize(self, _discard, self._tmp_path)
        _write_header(self._file)
        self._features = self._file.create_dataset(
            FEATURES_DATASET,
            shape=(0, *self._feature_shape),
            maxshape=(None, *self._feature_shape),
            dtype=self._dtype,
            chunks=chunks,
            compression=compression,
        )
        self._labels = self._file.create_dataset(
            LABELS_DATASET,
            shape=(0,),
            maxshape=(None,),
            dtype=_string_dtype(),
            chunks=(_LABELS_CHUNK,),
        )

    @property
    def n_samples(self) -> int:
        """Number of samples written so far."""
        return self._n

    def append(self, feature: np.ndarray, label: str) -> None:
        """Append a single sample.

        Parameters
        ----------
        feature : numpy.ndarray
            One sample, either ``feature_shape`` or ``(1, *feature_shape)``.
        label : str
            Its stimulus label.
        """
        feature = np.asarray(feature)
        if feature.shape == self._feature_shape:
            feature = feature[np.newaxis]
        self.extend(feature, [label])

    def extend(self, features: np.ndarray, labels: Sequence[str]) -> None:
        """Append several samples at once.

        Parameters
        ----------
        features : numpy.ndarray
            Samples of shape ``(n, *feature_shape)``.
        labels : sequence of str
            One label per sample.

        Raises
        ------
        ValueError
            If the feature shape does not match the writer's, or the label count
            does not match the sample count.
        RuntimeError
            If the writer is already closed.
        """
        if self._file is None:
            raise RuntimeError("FeatureWriter is closed")
        features = np.asarray(features)
        labels = list(labels)
        if features.shape[1:] != self._feature_shape:
            raise ValueError(
                "expected features of shape (n, {}), got {}".format(
                    ", ".join(str(s) for s in self._feature_shape), features.shape
                )
            )
        if len(labels) != features.shape[0]:
            raise ValueError(
                "got {} labels for {} samples".format(len(labels), features.shape[0])
            )
        # Check before writing anything, so a rejected batch leaves the file
        # exactly as it was.
        self._reject_duplicates(labels)
        if not labels:
            return

        new_n = self._n + len(labels)
        self._features.resize(new_n, axis=0)
        self._labels.resize(new_n, axis=0)
        self._features[self._n:new_n] = features.astype(self._dtype, copy=False)
        self._labels[self._n:new_n] = labels
        self._seen.update(labels)
        self._n = new_n

    def _reject_duplicates(self, labels: Sequence[str]) -> None:
        """Refuse labels repeated within this batch or already written."""
        within = _duplicates(labels)
        if within:
            raise ValueError(
                "duplicate labels in this batch: {}. Labels identify rows, so "
                "they must be unique.".format(
                    ", ".join(repr(d) for d in within[:3])
                )
            )
        already = [label for label in labels if label in self._seen]
        if already:
            raise ValueError(
                "labels already written to {}: {}. Labels identify rows, so "
                "they must be unique.".format(
                    self._path, ", ".join(repr(d) for d in already[:3])
                )
            )

    def close(self) -> None:
        """Finish the file and move it into place. Idempotent.

        This is the commit: call it only once the data is complete, because it
        publishes whatever has been written so far. To throw a partial file
        away, call :meth:`abort` instead. ``try: ... finally: writer.close()``
        is therefore wrong -- it would publish a truncated file -- so use the
        writer as a context manager, which routes to :meth:`abort` when the
        body raises.

        Raises
        ------
        RuntimeError
            If the writer was already aborted.
        FileExistsError
            If the target was published meanwhile; the writer is then aborted.
        """
        if self._aborted:
            raise RuntimeError(
                "FeatureWriter was aborted; nothing was written to {}".format(
                    self._path
                )
            )
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._tmp_path is not None:
            try:
                _promote(self._tmp_path, self._path, self._overwrite)
            except FileExistsError:
                self.abort()
                raise
            self._tmp_path = None
            self._finalizer.detach()  # the scratch file is now the output

    def abort(self) -> None:
        """Discard the partial file without touching the target path.

        Idempotent, and a no-op once :meth:`close` has published the file --
        it must never remove a published output.
        """
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._tmp_path is not None:
            self._finalizer()  # runs _discard exactly once
            self._tmp_path = None
            self._aborted = True

    def __enter__(self) -> "FeatureWriter":
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Publish the file on success, discard it if the block raised."""
        if self._tmp_path is None:
            return  # the body already called close() or abort() itself
        if exc_type is None:
            self.close()
        else:
            self.abort()


def convert_features_to_hdf5(
    src_dir: str,
    dst_dir: str,
    layers: Optional[Iterable[str]] = None,
    ext: str = "mat",
    key: str = "feat",
    overwrite: bool = False,
    batch_size: int = 64,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
    compression: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Convert a legacy per-stimulus feature directory to chunked HDF5.

    Reads `src_dir` through the same backend :class:`~bdpy.dataform.Features`
    uses for the legacy layout, so the output is by construction what the legacy
    reader sees. Samples are streamed in batches, so a layer is never held in
    memory in full.

    Parameters
    ----------
    src_dir : str
        Legacy directory, ``<src_dir>/<layer>/<label>.<ext>``.
    dst_dir : str
        Output directory; ``<dst_dir>/<layer>.h5`` is written per layer. Created
        if missing. Each layer is written atomically, so a failed conversion
        leaves no file behind and can simply be re-run.
    layers : iterable of str, optional
        Layers to convert. Defaults to every layer found in `src_dir`.
    ext : str, optional
        Extension of the legacy files (default: ``'mat'``).
    key : str, optional
        Variable name inside the legacy files (default: ``'feat'``).
    overwrite : bool, optional
        Overwrite an existing output file instead of skipping it
        (default: False). Without it, a layer that another process publishes
        while this one is converting it is skipped as well.
    batch_size : int, optional
        Number of stimulus files read per batch. Must be positive.
    target_chunk_bytes : int, optional
        Per-chunk byte budget.
    compression : str, optional
        h5py compression filter. ``None`` (default) stores uncompressed.
    verbose : bool, optional
        Print progress.

    Raises
    ------
    ValueError
        If `batch_size` is not positive.
    KeyError
        If a requested layer is not present in `src_dir`.
    """
    if batch_size < 1:
        # range(0, n, -1) is empty, so a non-positive batch size would write no
        # samples at all and then publish that empty file as a finished layer.
        raise ValueError("batch_size must be positive, got {}".format(batch_size))

    store = MatFeatureStore(src_dir, ext=ext, key=key)
    selected = list(store.layers) if layers is None else list(layers)
    missing = [layer for layer in selected if layer not in store.layers]
    if missing:
        raise KeyError(
            "Layer(s) {} not found in {}".format(", ".join(missing), src_dir)
        )

    os.makedirs(dst_dir, exist_ok=True)
    all_labels = store.labels

    for layer in selected:
        out_path = os.path.join(dst_dir, layer + "." + HDF5_EXT)
        if os.path.exists(out_path) and not overwrite:
            if verbose:
                print("{} already exists. Skipped.".format(out_path))
            continue

        full_shape = store.shape(layer)
        # The context manager is load-bearing: if a batch fails, it discards the
        # partial file instead of publishing it. A published partial file would
        # be taken for a finished one by the skip check above, and the layer
        # would stay silently truncated across re-runs.
        try:
            with FeatureWriter(
                out_path,
                feature_shape=full_shape[1:],
                dtype=store.dtype(layer),
                n_samples=full_shape[0],
                target_chunk_bytes=target_chunk_bytes,
                compression=compression,
                overwrite=overwrite,
            ) as writer:
                for start in range(0, len(all_labels), batch_size):
                    batch = all_labels[start:start + batch_size]
                    writer.extend(store.read(layer, batch), batch)
        except FileExistsError:
            # Only reachable without overwrite: another process published this
            # layer while we were converting it. Treat it like the check above;
            # the writer has already discarded our scratch file.
            if verbose:
                print("{} already exists. Skipped.".format(out_path))
            continue

        if verbose:
            print("Saved {} ({} samples).".format(out_path, full_shape[0]))


def _makedirs_for(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
