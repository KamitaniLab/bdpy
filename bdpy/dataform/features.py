"""DNN features class

This file is a part of BdPy.
"""


from __future__ import print_function

__all__ = ['DecodedFeatures', 'Features', 'save_feature']

import glob
import os
import pickle
import sqlite3
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import hdf5storage
import numpy as np

from . import _mat_v73
from ._feature_store import (
    FeatureSlice,
    FeatureStore,
    HDF5FeatureStore,
    MatFeatureStore,
    _determine_num_parallel,
    _load_array_with_key,
    detect_format,
)

# Deprecation notice emitted by the MATLAB-compatible write path. The actual
# switch to bdpy-native plain HDF5 (and the drop of the hdf5storage write
# dependency) is implemented on the refactor/drop-hdf5storage-write branch.
_MATLAB_WRITE_FUTURE_WARNING = (
    "Writing MATLAB-compatible v7.3 .mat files is deprecated and will change "
    "in a future release: bdpy will write bdpy-native plain HDF5 instead. "
    "Newly written files will no longer be guaranteed to be readable by "
    "MATLAB's load(). Reading existing hdf5storage / MATLAB v7.3 files remains "
    "supported."
)


class Features(object):
    """DNN features class.

    Reads a collection of DNN features from disk. Two on-disk layouts are
    supported and behave identically through this API:

    - the legacy per-stimulus layout, ``<dpath>/<layer>/<label>.mat``;
    - chunked HDF5 storage, ``<dpath>/<layer>.h5`` (see
      :mod:`bdpy.dataform.feature_hdf5`), which additionally supports partial
      reads along the feature axes without loading a whole layer.

    Parameters
    ----------
    dpath: str or list
        (List of) DNN feature directory(ies)
    ext: str
        DNN feature file extension of the legacy per-stimulus layout
        (default: mat). Ignored by chunked HDF5 storage.
    feature_index: str, optional
        Path to a ``.mat`` file holding a per-layer unit index under the key
        ``index``. When given, features are flattened and restricted to those
        units.
    format: str
        Storage format: ``'auto'`` (default), ``'mat'`` or ``'hdf5'``. With
        ``'auto'`` each directory is inspected independently, so a mix of
        layouts across `dpath` entries is fine.

    Attributes
    ----------
    labels: list
        List of stimulus labels
    index: list
        List of stimulus index (one-based)
    layers: list
        List of DNN layers
    """

    def __init__(
            self, dpath: Union[str, List[str]] = [],
            ext: str = 'mat', feature_index: Optional[str] = None,
            format: str = 'auto'
        ):
        if not isinstance(dpath, list):
            dpath = [dpath]
        self.__dpath = dpath
        self.__format = format

        self.__stores: List[FeatureStore] = []
        self.__label_store: Dict[str, FeatureStore] = {}  # label -> owning store
        self.__labels: List[str] = []  # Stimulus labels
        self.__index: List[int] = []  # Stimulus index (one-based)
        self.__layers: List[str] = []  # DNN layers
        self.__collect_features(ext=ext)

        self.__c_feature_name: Optional[str] = None  # Loaded layer
        self.__features: Optional[np.ndarray] = None  # Loaded features
        self.__feature_index = None   # Indexes of loaded features
        # NOTE: type of self.__feature_index is ambiguous

        if feature_index is not None:
            if not os.path.exists(feature_index):
                raise RuntimeError('%s do not exist' % feature_index)
            self.__feat_index_table = _mat_v73.loadmat_key(feature_index, 'index')
            # NOTE: type of self.__feature_index_table is ambiguous
        else:
            self.__feat_index_table = None

        self.__statistics = {}
        for fdir in self.__dpath:
            stat_file = os.path.join(fdir, 'statistics.pkl')
            if os.path.exists(stat_file):
                with open(stat_file, 'rb') as f:
                    feat_stat = pickle.load(f)
                self.__statistics.update(feat_stat)

    @property
    def labels(self):
        return self.__labels

    @property
    def index(self):
        return self.__index

    @property
    def layers(self):
        return self.__layers

    @property
    def feature_index(self):
        return self.__feature_index

    def shape(self, layer: str) -> Tuple[int, ...]:
        """Return the full shape of `layer` without reading the features.

        Parameters
        ----------
        layer: str
            DNN layer

        Returns
        -------
        tuple of int
            ``(n_samples, *feature_shape)``
        """
        feature_shape = self.__stores[0].shape(layer)[1:]
        return (len(self.__labels), *feature_shape)

    def get(
            self, layer: str, label: Union[str, List[str], None] = None,
            feature_slice: FeatureSlice = None
        ) -> np.ndarray:
        """Return features in `layer`.

        Parameters
        ----------
        layer: str
            DNN layer
        label: str or list
            Sample label(s). Rows come back in the order given.
        feature_slice: slice, int, Ellipsis or tuple of those, optional
            Index applied to the feature axes (axes 1 and up), as produced by
            ``numpy.s_[...]``. Basic forward indexing only: slices with a
            positive step and integer bounds, integers, and at most one
            ``Ellipsis``. Fancy indexing, a negative step, a non-integer slice
            bound, booleans, and ``numpy.newaxis`` *inside a tuple* raise
            ``ValueError``. A bare ``numpy.newaxis`` is ``None``, the "no
            slice" default, so it reads the whole feature tensor. With chunked
            HDF5 storage this is a genuine partial read; with the legacy layout
            the files are loaded in full and then sliced.

        Returns
        -------
        numpy.ndarray, shape=(n_samples, shape_layers)
            DNN features

        Examples
        --------
        >>> features.get('conv5')                                  # doctest: +SKIP
        >>> features.get('conv5', label=['img0001', 'img0002'])     # doctest: +SKIP
        >>> features.get('conv5', feature_slice=np.s_[128:256])     # doctest: +SKIP

        Raises
        ------
        ValueError
            If `feature_slice` is combined with a unit index (`feature_index`).
            The index addresses the flattened *full* feature space, so it is
            meaningless against an already-sliced array.
        """
        if feature_slice is not None and self.__feat_index_table is not None:
            # The unit index addresses the flattened full feature space, while
            # feature_slice has already narrowed it. Applying one to the other
            # would silently select the wrong units (or raise IndexError),
            # so refuse the combination instead of guessing.
            raise ValueError(
                'feature_slice cannot be combined with feature_index: the unit '
                'index addresses the full feature space, not a slice of it. '
                'Call get() without feature_slice and slice the result, or '
                'construct Features without feature_index.'
            )

        if label is None and feature_slice is None:
            return self.get_features(layer)

        if label is None:
            labels = self.__labels
        elif isinstance(label, str):
            labels = [label]
        else:
            labels = list(label)

        features = self.__read(layer, labels, feature_slice)
        return self.__apply_feature_index(features, layer)

    def iter_chunks(
            self, layer: str, label: Union[str, List[str], None] = None,
            axis: int = 1, size: Optional[int] = None
        ) -> Iterator[Tuple[slice, np.ndarray]]:
        """Iterate over `layer` in slabs along `axis`.

        This is the streaming counterpart of :meth:`get`: with chunked HDF5
        storage it holds only one slab at a time, so a layer far larger than RAM
        can be processed end to end.

        There is no `feature_slice` here on purpose -- slice the blocks as they
        come out instead. Composing an arbitrary slice with the per-slab slice
        means reimplementing NumPy's index arithmetic, which is more generality
        than reading a layer in slabs needs.

        Parameters
        ----------
        layer: str
            DNN layer
        label: str or list, optional
            Sample label(s). ``None`` iterates over every label.
        axis: int
            Axis of the selected array to iterate over. Axis 0 is the sample
            axis; the default, axis 1, is the outermost feature axis.
        size: int, optional
            Elements per slab. Defaults to the on-disk chunk extent along
            `axis`, so that each element is read exactly once.

        Notes
        -----
        The one-slab-at-a-time guarantee is a property of chunked HDF5 storage
        and of `axis=0` on any layout. The legacy `.mat` layout has no file
        boundary on the feature axes, so iterating one reads the selection once
        -- the same peak as a single :meth:`get` -- and yields views into it,
        which keep that array alive for as long as a block is held. It is still
        read once rather than once per slab.

        The guarantee also assumes the requested labels all live in a single
        feature directory, which is the usual case. When they are spread across
        several `dpath` entries, no single store can stream them, so the
        selection is read in full and then split.

        Yields
        ------
        tuple of (slice, numpy.ndarray)
            The slice applied to `axis`, and the corresponding slab.

        Raises
        ------
        ValueError
            If a unit index (`feature_index`) is in use, which flattens the
            feature axes and so has no meaningful per-axis iteration; if `axis`
            is out of range; or if `size` is given and is not positive.

        Examples
        --------
        >>> for sl, block in features.iter_chunks('conv5'):  # doctest: +SKIP
        ...     out[:, sl] = transform(block)
        """
        if self.__feat_index_table is not None:
            raise ValueError(
                'iter_chunks is not supported together with feature_index, '
                'which flattens the feature axes'
            )

        if label is None:
            labels = self.__labels
        elif isinstance(label, str):
            labels = [label]
        else:
            labels = list(label)

        # Validate here rather than in the store, so that both dispatch paths
        # below reject the same inputs the same way. The multi-store fallback
        # does not go through FeatureStore.iter_chunks and would otherwise
        # silently yield nothing for size < 1, and raise IndexError rather than
        # ValueError for an out-of-range axis.
        ndim = len(self.shape(layer))
        if axis < 0:
            axis += ndim
        if not 0 <= axis < ndim:
            raise ValueError(
                'axis {} is out of range for a {}-dimensional selection'.format(
                    axis, ndim
                )
            )
        if size is not None and size < 1:
            raise ValueError('size must be positive, got {}'.format(size))

        store = self.__store_for(labels)
        if store is not None:
            yield from store.iter_chunks(layer, labels, axis=axis, size=size)
            return

        # Labels span several directories, so no single store can stream them.
        # Fall back to slicing a full read, which still yields the same blocks.
        features = self.__read(layer, labels)
        length = features.shape[axis]
        if size is None:
            size = length
        for start in range(0, length, size):
            sl = slice(start, min(start + size, length))
            index: List[Any] = [slice(None)] * features.ndim
            index[axis] = sl
            yield sl, features[tuple(index)]

    def statistic(self, statistic: str = 'mean', layer: Optional[str] = None):
        # NOTE: return type is ambiguous. currently, it is inferred as Unkown | Any

        if statistic == 'std':
            statistic = 'std, ddof=1'

        k = (statistic, layer)
        if k in self.__statistics:
            s = self.__statistics[k]
        else:
            f = self.get(layer)  # NOTE: here, layer could be None. It will raise RuntimeError.

            if statistic == 'mean':
                s = np.mean(f, axis=0)[np.newaxis, :]
            elif statistic == 'std, ddof=1':
                s = np.std(f, axis=0, ddof=1)[np.newaxis, :]
            elif statistic == 'std, ddof=0':
                s = np.std(f, axis=0, ddof=0)[np.newaxis, :]
            else:
                raise ValueError('Unknown statistics: {}'.format(statistic))

            self.__statistics.update({k: s})

        if self.__feat_index_table is not None:
            # Select features by index
            self.__feature_index = self.__feat_index_table[layer]
            assert isinstance(self.__features, np.ndarray)
            n_sample = self.__features.shape[0]  # self.__features could be None
            n_feat = np.array(self.__features.shape[1:]).prod()

            s = s.reshape([n_sample, n_feat], order='C')[:, self.__feature_index]

        return s

    def get_features(self, layer: str) -> np.ndarray:
        """Return features in `layer`.

        Parameters
        ----------
        layer: str
            DNN layer

        Returns
        -------
        numpy.ndarray, shape=(n_samples, shape_layers)
            DNN features
        """
        if layer == self.__c_feature_name:
            assert isinstance(self.__features, np.ndarray)
            return self.__features  # self.__features could be None

        self.__features = self.__read(layer, self.__labels)
        self.__c_feature_name = layer
        self.__features = self.__apply_feature_index(self.__features, layer)

        return self.__features

    def __read(
            self, layer: str, labels: Sequence[str],
            feature_slice: FeatureSlice = None
        ) -> np.ndarray:
        """Read `labels` from `layer`, dispatching to the owning store(s)."""
        store = self.__store_for(labels)
        if store is not None:
            return store.read(layer, labels, feature_slice)
        # Labels are spread over several directories: read each run of
        # consecutive labels from its own store, preserving the caller's order.
        blocks: List[np.ndarray] = []
        run: List[str] = []
        run_store: Optional[FeatureStore] = None
        for label in labels:
            owner = self.__label_store[label]
            if owner is not run_store and run:
                assert run_store is not None
                blocks.append(run_store.read(layer, run, feature_slice))
                run = []
            run_store = owner
            run.append(label)
        if run:
            assert run_store is not None
            blocks.append(run_store.read(layer, run, feature_slice))
        return np.concatenate(blocks, axis=0)

    def __store_for(self, labels: Sequence[str]) -> Optional[FeatureStore]:
        """The single store owning every label, or None if they are spread out."""
        if len(self.__stores) == 1:
            return self.__stores[0]
        owners = {id(self.__label_store[label]) for label in labels}
        if len(owners) == 1:
            return self.__label_store[labels[0]]
        return None

    def __apply_feature_index(self, features: np.ndarray, layer: str) -> np.ndarray:
        if self.__feat_index_table is None:
            return features
        # Select features by index
        self.__feature_index = self.__feat_index_table[layer]
        n_sample = features.shape[0]
        n_feat = np.array(features.shape[1:]).prod()
        return features.reshape([n_sample, n_feat], order='C')[:, self.__feature_index]

    def __collect_features(self, ext='mat'):
        for dpath in self.__dpath:
            store = self.__make_store(dpath, ext)
            if self.__layers and store.layers != self.__layers:
                raise RuntimeError('Invalid layers in %s' % dpath)
            self.__layers = store.layers
            self.__stores.append(store)
            for label in store.labels:
                self.__label_store[label] = store
            self.__labels += store.labels

        # NOTE: type incompatibility here. Is it OK to cast to list?
        self.__index = np.arange(len(self.__labels)) + 1

        return None

    def __make_store(self, dpath: str, ext: str) -> FeatureStore:
        fmt = self.__format
        if fmt == 'auto':
            fmt = detect_format(dpath, ext=ext)
        if fmt == 'hdf5':
            return HDF5FeatureStore(dpath)
        if fmt == 'mat':
            return MatFeatureStore(dpath, ext=ext)
        raise ValueError(
            "Unknown feature storage format {!r}; expected "
            "'auto', 'mat' or 'hdf5'".format(fmt)
        )


class DecodedFeatures(object):
    """Decoded features class.

    Parameters
    ----------
    path: str
        Path to the decoded feature directory
    """

    def __init__(
            self, path: str, keys: Optional[List[str]] = None, file_ext: str = 'mat',
            file_key: str = 'feat', squeeze: bool = False):

        self.__path = path          # Path to decoded feature directory
        self.__keys = keys          # Keys
        self.__file_ext = file_ext  # Decoded feature file extension
        self.__file_key = file_key  # Decoded feature data key (FIXME)
        self.__squeeze = squeeze    # Whether squeeze the output array or not

        self.__db = self.__parse_dir(self.__path, self.__keys)

        stat_file = os.path.join(self.__path, 'statistics.pkl')

        if os.path.exists(stat_file):
            with open(stat_file, 'rb') as f:
                self.__statistics = pickle.load(f)
        else:
            self.__statistics = {}

    @property
    def layers(self):
        return self.__db.get_available_values('layer')

    @property
    def subjects(self):
        return self.__db.get_available_values('subject')

    @property
    def rois(self):
        return self.__db.get_available_values('roi')

    @property
    def folds(self):
        return self.__db.get_available_values('fold')

    @property
    def labels(self):
        return self.__db.get_available_values('label')

    @property
    def selected_layer(self):
        return self.__db.get_selected_values('layer')

    @property
    def selected_subject(self):
        return self.__db.get_selected_values('subject')

    @property
    def selected_roi(self):
        return self.__db.get_selected_values('roi')

    @property
    def selected_fold(self):
        return self.__db.get_selected_values('fold')

    @property
    def selected_label(self):
        return self.__db.get_selected_values('label')

    def get(self, layer=None, subject=None, roi=None, fold=None, label=None, image=None):
        """Returns decoded features as an array."""
        if image is not None:
            if label is None:
                warnings.warn('`image` will be deprecated.')
                label = image
            else:
                warnings.warn('`image` will be deprecated and overwritten by `label`.')

        files = self.__db.get_file(
            layer=layer,
            subject=subject,
            roi=roi,
            fold=fold,
            label=label
        )

        if len(files) == 0:
            raise RuntimeError('No decoded feature found')

        features: np.ndarray
        num_files = len(files)
        num_parallel = _determine_num_parallel(num_files)
        if num_parallel == 1:
            features = np.concatenate(list(map(partial(_load_array_with_key, self.__file_key), files)), axis=0)
        else:
            with Pool(processes=num_parallel) as pool:
                features = np.concatenate(pool.map(partial(_load_array_with_key, self.__file_key), files), axis=0)

        if self.__squeeze:
            features = np.squeeze(features)

        return features

    def statistic(self, statistic='mean', layer=None, subject=None, roi=None, fold=None):

        if statistic == 'std':
            statistic = 'std, ddof=1'

        k = (statistic, layer, subject, roi, fold)
        if k in self.__statistics:
            s = self.__statistics[k]
        else:
            f = self.get(layer=layer, subject=subject, roi=roi, fold=fold)

            if statistic == 'mean':
                s = np.mean(f, axis=0)[np.newaxis, :]
            elif statistic == 'std, ddof=1':
                s = np.std(f, axis=0, ddof=1)[np.newaxis, :]
            elif statistic == 'std, ddof=0':
                s = np.std(f, axis=0, ddof=0)[np.newaxis, :]
            else:
                raise ValueError('Unknown statistics: {}'.format(statistic))

            self.__statistics.update({k: s})

        return s

    def __parse_dir(self, path: str, keys: Optional[List[str]] = None) -> 'FileDatabase':
        # TODO: refactoring
        if keys is None:
            files = glob.glob(os.path.join(path, '*', '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
            keys = ['layer', 'subject', 'roi', 'fold', 'label']
            if len(files) == 0:
                files = glob.glob(os.path.join(path, '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
                keys = ['layer', 'subject', 'roi', 'label']
            if len(files) == 0:
                files = glob.glob(os.path.join(path, '*', '*', '*', '*.' + self.__file_ext))
                keys = ['layer', 'subject', 'roi', 'label']
        elif len(keys) == 4:
            # <layer>/<subject>/<roi>/<label>
            files = glob.glob(os.path.join(path, '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
            if len(files) == 0:
                files = glob.glob(os.path.join(path, '*', '*', '*', '*.' + self.__file_ext))
            if len(files) == 0:
                raise RuntimeError('Decoded features not found')
        elif len(keys) == 5:
            # <layer>/<subject>/<roi>/<fold>/<label>
            files = glob.glob(os.path.join(path, '*', '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
        else:
            raise ValueError('Invalid keys')

        if len(files) == 0:
            raise RuntimeError('Decoded features not found')

        print('Found {} decoded features in {}'.format(len(files), self.__path))

        self.__keys = keys

        db = FileDatabase(keys)

        # TODO: performance improvement
        for file in files:
            # FXIME: "decoded_features"
            k = {
                k: os.path.splitext(file.replace('/decoded_features/', '/'))[0].split('/')[i - len(keys)]
                for i, k in enumerate(keys)
            }
            db.add_file(file, **k)

        return db


class FileDatabase(object):
    def __init__(self, keys: List[str]):
        self.__keys = keys

        self.__res: Optional[List[Any]] = None

        self.__con = sqlite3.connect(':memory:')
        self.__cursor = self.__con.cursor()

        self.__cursor.execute(
            '''
            CREATE TABLE files (
            {},
            path TEXT,
            UNIQUE ({})
            )
            '''.format(
                (', ').join([s + ' TEXT' for s in keys]),
                (', ').join(keys)
            )
        )

    def add_file(self, path: str, **kargs):
        key_list = ', '.join(kargs) + ', path'
        val_list = ', '.join(['"{}"'.format(s) for s in kargs.values()]) + ', "{}"'.format(path)
        self.__cursor.execute('INSERT INTO files({}) VALUES ({})'.format(key_list, val_list))

    def get_file(self, **kargs):
        where = ' AND '.join(['{} = "{}"'.format(k, v) for k, v in kargs.items() if k in self.__keys and v is not None])
        self.__cursor.execute('SELECT * FROM files WHERE {}'.format(where))
        self.__res = self.__cursor.fetchall()
        return [a[-1] for a in self.__res]

    def get_available_values(self, key: str):
        if key not in self.__keys:
            return None
        self.__cursor.execute('SELECT DISTINCT {} FROM files'.format(key))
        return [a[0] for a in self.__cursor.fetchall()]

    def get_selected_values(self, key):
        if key not in self.__keys:
            return None
        # NOTE: self.__res could be None
        # This design forces users to call get_file() before get_selected_values()
        return [a[self.__keys.index(key)] for a in self.__res]

    def show(self):
        self.__cursor.execute('SELECT * FROM files')
        print(self.__cursor.fetchall())


def save_feature(feature: np.ndarray, base_dir: str, layer: str, label: str, verbose: bool = False):
    """
    Save features.

    Parameters
    ----------
    feature: np.ndarray
    base_dir: str
    layer: str
    label: str
    verbose: bool (default: False)

    Returns
    -------
    None
    """
    save_dir = os.path.join(base_dir, layer)
    os.makedirs(save_dir, exist_ok=True)

    save_file = os.path.join(save_dir, label + '.mat')
    if os.path.exists(save_file):
        if verbose:
            print(f'{save_file} already exists. Skipped.')
        return None

    warnings.warn(_MATLAB_WRITE_FUTURE_WARNING, FutureWarning, stacklevel=2)
    hdf5storage.savemat(save_file, {'feat': feature})
    if verbose:
        print(f'Saved {save_file}.')

    return None
