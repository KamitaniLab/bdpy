"""Utilities for Bdpy dataformat."""

from typing import List, Union

import numpy as np

from bdpy.dataform import Features


def get_multi_features(features: List[Features], layer: str, labels: Union[List[str], np.ndarray]) -> np.ndarray:
    """Load features from multiple Features."""
    y_list = []
    for label in labels:
        for feat in features:
            if label not in feat.labels:
                continue
            f = feat.get(layer=layer, label=label)
            y_list.append(f)
    return np.vstack(y_list)
