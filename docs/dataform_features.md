# Features and DecodedFeatures

bdpy provides classes to handle DNN's (true) features and decoded features: `dataform.Features` and `dataform.DecodedFeatures`.

## Basic usage

``` python
from bdpy.dataform import Features, DecodedFeatures


## Initialize

features = Features('/path/to/features/dir')

decoded_features = DecodedFeatures('/path/to/decoded/features/dir')

## Get features as an array

feat = features.get(layer='conv1')

decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC', label='stimulus-0001)  # Decoded features for specified sample (label)
decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC')                        # Decoded features from all avaiable samples

# Decoded features with CV
decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC', fold='cv_fold1)

## List labels

feat_labels = features.labels

decfeat_labels = decoded_features.labels          # All available labels
decfeat_labels = decoded_features.selected_label  # Labels assigned to decoded features previously obtained by `get` method
```

## Feature statistics

``` python
features.statistic('mean', layer='fc8')
features.statistic('std', layer='fc8')          # Default ddof = 1
features.statistic('std, ddof=0', layer='fc8')

decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC')
decoded_features.statistic('std', layer='fc8', subject='sub-01', roi='VC')          # Default ddof = 1
decoded_features.statistic('std, ddof=0', layer='fc8', subject='sub-01', roi='VC')

# Decoded features with CV
decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC', fold='cv_fold1')  # Mean within the specified fold
decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC')

# If `fold` is omitted for CV decoded features, decoded features are pooled across add CV folds and then the statistics are calculated.

```

## Chunked HDF5 feature storage

The layout above stores one file per stimulus, so the only file boundary is the
sample axis: reading a few channels of a layer still costs a full read of every
stimulus. For large DNN features, `Features` can instead read **chunked HDF5**
storage, one file per layer:

```
features/
  conv1_1.h5    # /features (n_stimuli, *feature_shape) + /labels (n_stimuli,)
  conv2_1.h5
  fc8.h5
```

`/features` is explicitly chunked along both the sample axis and the outermost
feature axis, so a slice reads only the chunks it covers.

### Reading

Nothing changes for existing code -- `Features` detects the layout per
directory, and `.mat` trees keep working exactly as before:

``` python
features = Features('/path/to/features')   # either layout
feat = features.get(layer='conv5')
```

Detection keys off the files that are actually present, so an unrelated
subdirectory next to `<layer>.h5` files does not make a directory look legacy.
A directory holding *both* layouts is ambiguous and raises rather than picking
one silently; pass `format='mat'` or `format='hdf5'` to resolve it.

### Partial reads

`feature_slice` selects along the feature axes (axis 1 and up). On chunked HDF5
this is a real partial read; on the legacy layout the files are loaded in full
and then sliced, so the same code works on both.

``` python
import numpy as np

# Channels 128-255 only, without loading the rest of the layer
feat = features.get(layer='conv5', feature_slice=np.s_[128:256])

# Combine with label selection; rows come back in the order given
feat = features.get(
    layer='conv5',
    label=['stimulus-0003', 'stimulus-0001'],
    feature_slice=np.s_[128:256],
)

# NOTE: feature_slice cannot be combined with a unit index (`feature_index`).
# The index addresses the flattened full feature space, so applying it to an
# already-sliced array would select the wrong units; the combination raises
# ValueError instead.
```

`feature_slice` is **basic forward indexing**: slices with a positive step
(negative `start`/`stop` are fine), integers, a single `Ellipsis`, and tuples of
those. Anything else -- a negative step, a non-integer slice bound, fancy
indexing with a list or array, booleans, or `np.newaxis` *inside a tuple* --
raises `ValueError` on *both* backends. (A bare `np.newaxis` is simply `None`,
which is the "no slice" default, so it reads the whole feature tensor rather
than raising.) The restriction is what lets the two layouts mean the same thing
by the same index; read without `feature_slice` and index the result with NumPy
when you need more.

``` python
features.get('conv5', feature_slice=np.s_[128:256])     # ok
features.get('conv5', feature_slice=np.s_[8:16, 1:4])   # ok
features.get('conv5', feature_slice=np.s_[::-1])        # ValueError
features.get('conv5', feature_slice=np.s_[[3, 1, 7]])   # ValueError
features.get('conv5', feature_slice=slice(1.5, 3))      # ValueError

# Full shape without reading anything
n_stimuli, *feature_shape = features.shape('conv5')
```

To stream a layer that does not fit in memory, iterate. `iter_chunks` yields
`(slice, block)` so results can be placed back without tracking offsets, and by
default uses the on-disk chunk extent, so every element is read exactly once:

``` python
out = np.empty(features.shape('conv5'))
for sl, block in features.iter_chunks('conv5', axis=1):
    out[:, sl] = transform(block)
```

`axis=0` iterates over stimuli instead of features. `iter_chunks` takes no
`feature_slice` -- slice each `block` as it comes out instead.

Only one slab is resident at a time for chunked HDF5, and for `axis=0` on either
layout. On the legacy `.mat` layout there is no file boundary on the feature
axes, so iterating one reads the selection once -- the same peak as a single
`get()` -- and yields views into it, which keep that array alive for as long as a
block is held. It is still read once rather than once per slab, which is what
makes `iter_chunks` usable on that layout at all. The same full-read fallback
applies when the requested labels are spread across several `dpath` entries,
since no single store can stream them.

### Writing

Write a whole layer at once:

``` python
from bdpy.dataform import save_features

save_features('features/conv5.h5', array, labels)
```

Writing never clobbers: an existing file raises `FileExistsError` unless you pass
`overwrite=True`. Writes are also atomic -- the file is built in a hidden
`.<name>.<id>.partial` sibling and moved into place only once it is complete --
so a failed write leaves nothing behind and can simply be re-run, and an
overwrite keeps the old file readable until the moment it is replaced. (A
process killed outright cannot clean up after itself, so a stray `.partial` file
may be left for you to delete.)

Or incrementally, which is what feature extraction needs since it produces one
stimulus at a time:

``` python
from bdpy.dataform import FeatureWriter

with FeatureWriter('features/conv5.h5', feature_shape=(256, 13, 13),
                   dtype=np.float32) as writer:
    for label, feature in extract():
        writer.append(feature, label)
```

### Migrating existing features

``` python
from bdpy.dataform import convert_features_to_hdf5

convert_features_to_hdf5('/path/to/features_mat', '/path/to/features_h5')
```

The converter reads through the same backend `Features` uses for the legacy
layout, and streams in batches, so a layer is never held in memory in full.
Existing `<layer>.h5` files are skipped unless `overwrite=True`, and a layer that
fails to convert leaves no file behind, so re-running picks up where it stopped.

All layers in a directory must hold the same stimulus labels in the same order,
and labels must be unique within a layer; both are rejected when the directory
is opened, as they are for the legacy layout (where the file name *is* the
label, so duplicates cannot arise). Writing a duplicate label raises too.

### Chunk shape

Chunk shape, not the file format alone, is what decides the cost of a partial
read: HDF5 reads whole chunks even when only a few of their elements are
selected. By default bdpy derives it from a 1 MiB budget, splitting it between
the sample axis and the outermost feature axis and keeping trailing spatial axes
whole -- e.g. `(1200, 256, 13, 13)` float32 becomes chunks of
`(39, 37, 13, 13)`. Extents are chosen to divide each axis as evenly as
possible, so HDF5 does not pad the edge chunks.

That means reading a *single* stimulus costs one chunk row (~39 stimuli of I/O),
which matters for `FeaturesDataset`-style per-sample access. Tune it if your
access pattern is skewed:

``` python
save_features(path, array, labels, target_chunk_bytes=256 * 1024)  # smaller chunks
save_features(path, array, labels, chunks=(1, 256, 13, 13))        # per-stimulus
```

Compression is off by default, since every compressed chunk costs a
decompression on the way out. Note that this makes the new format *larger* on
disk than a legacy `.mat` tree, whose files `hdf5storage` compresses by default.
Pass `compression='gzip'` or `compression='lzf'` when size matters more than
read speed.

### Format

Files are marked with root attributes `bdpy_format="features"` and
`bdpy_format_version=1`, and are rejected on read if those are missing or
newer than the running bdpy understands.
