from __future__ import annotations

from typing import Iterable, Callable, Dict

from pathlib import Path

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from bdpy.dataform import DecodedFeatures, Features


_FeatureTypeNP = Dict[str, np.ndarray]


def _removesuffix(s: str, suffix: str) -> str:
    """Remove suffix from string.

    Note
    ----
    This function is available from Python 3.9 as `str.removesuffix`. We can
    remove this function when we drop support for Python 3.8.

    Parameters
    ----------
    s : str
        String.
    suffix : str
        Suffix to remove.

    Returns
    -------
    str
        String without suffix.
    """
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s[:]


class FeaturesDataset(Dataset):
    """Dataset of features.

    Parameters
    ----------
    root_path : str | Path
        Path to the root directory of features.
    layer_path_names : Iterable[str]
        List of layer path names. Each layer path name is used to get features
        from the root directory so that the layer path name must be a part of
        the path to the layer.
    stimulus_names : list[str], optional
        List of stimulus names. If None, all stimulus names are used.
    transform : callable, optional
        Callable object which is used to transform features. The callable object
        must take a dict of features and return a dict of features.
    """

    def __init__(
        self,
        root_path: str | Path,
        layer_path_names: Iterable[str],
        stimulus_names: list[str] | None = None,
        transform: Callable[[_FeatureTypeNP], _FeatureTypeNP] | None = None,
    ):
        self._features_store = Features(Path(root_path).as_posix())
        self._layer_path_names = layer_path_names
        if stimulus_names is None:
            stimulus_names = self._features_store.labels
        self._stimulus_names = stimulus_names
        self._transform = transform

    def __len__(self) -> int:
        return len(self._stimulus_names)

    def __getitem__(self, index: int) -> _FeatureTypeNP:
        stimulus_name = self._stimulus_names[index]
        features = {}
        for layer_path_name in self._layer_path_names:
            feature = self._features_store.get(
                layer=layer_path_name, label=stimulus_name
            )
            feature = feature[0]  # NOTE: remove batch axis
            features[layer_path_name] = feature
        if self._transform is not None:
            features = self._transform(features)
        return features


class DecodedFeaturesDataset(Dataset):
    """Dataset of decoded features.

    Parameters
    ----------
    root_path : str | Path
        Path to the root directory of decoded features.
    layer_path_names : Iterable[str]
        List of layer path names. Each layer path name is used to get features
        from the root directory so that the layer path name must be a part of
        the path to the layer.
    subject_id : str
        ID of the subject.
    roi : str
        ROI name.
    stimulus_names : list[str], optional
        List of stimulus names. If None, all stimulus names are used.
    transform : callable, optional
        Callable object which is used to transform features. The callable object
        must take a dict of features and return a dict of features.
    """

    def __init__(
        self,
        root_path: str | Path,
        layer_path_names: Iterable[str],
        subject_id: str,
        roi: str,
        stimulus_names: list[str] | None = None,
        transform: Callable[[_FeatureTypeNP], _FeatureTypeNP] | None = None,
    ):
        self._decoded_features_store = DecodedFeatures(Path(root_path).as_posix())
        self._layer_path_names = layer_path_names
        self._subject_id = subject_id
        self._roi = roi
        if stimulus_names is None:
            stimulus_names = self._decoded_features_store.labels
            assert stimulus_names is not None
        self._stimulus_names = stimulus_names
        self._transform = transform

    def __len__(self) -> int:
        return len(self._stimulus_names)

    def __getitem__(self, index: int) -> _FeatureTypeNP:
        stimulus_name = self._stimulus_names[index]
        decoded_features = {}
        for layer_path_name in self._layer_path_names:
            decoded_feature = self._decoded_features_store.get(
                layer=layer_path_name,
                label=stimulus_name,
                subject=self._subject_id,
                roi=self._roi,
            )
            decoded_feature = decoded_feature[0]  # NOTE: remove batch axis
            decoded_features[layer_path_name] = decoded_feature
        if self._transform is not None:
            decoded_features = self._transform(decoded_features)
        return decoded_features


class ImageDataset(Dataset):
    """Dataset of images.

    Parameters
    ----------
    root_path : str | Path
        Path to the root directory of images.
    stimulus_names : list[str], optional
        List of stimulus names. If None, all stimulus names are used.
    extension : str, optional
        Extension of the image files.
    """

    def __init__(
        self,
        root_path: str | Path,
        stimulus_names: list[str] | None = None,
        extension: str = "jpg",
    ):
        self.root_path = root_path
        if stimulus_names is None:
            stimulus_names = [
                _removesuffix(path.name, "." + extension)
                for path in Path(root_path).glob(f"*{extension}")
            ]
        self._stimulus_names = stimulus_names
        self._extension = extension

    def __len__(self):
        return len(self._stimulus_names)

    def __getitem__(self, index: int):
        stimulus_name = self._stimulus_names[index]
        image = Image.open(Path(self.root_path) / f"{stimulus_name}.{self._extension}")
        image = image.convert("RGB")
        return np.array(image) / 255.0, stimulus_name


class RenameFeatureKeys:
    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def __call__(self, features: _FeatureTypeNP) -> _FeatureTypeNP:
        return {self._mapping.get(key, key): value for key, value in features.items()}
