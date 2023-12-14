from __future__ import annotations

import warnings

import numpy as np
import torch

from .core import Domain, IrreversibleDomain, ComposedDomain


def _bgr2rgb(images: torch.Tensor) -> torch.Tensor:
    """Convert images from BGR to RGB"""
    return images[:, [2, 1, 0], ...]


def _rgb2bgr(images: torch.Tensor) -> torch.Tensor:
    """Convert images from RGB to BGR"""
    return images[:, [2, 1, 0], ...]


def _to_channel_first(images: torch.Tensor) -> torch.Tensor:
    """Convert images from channel last to channel first"""
    return images.permute(0, 3, 1, 2)


def _to_channel_last(images: torch.Tensor) -> torch.Tensor:
    """Convert images from channel first to channel last"""
    return images.permute(0, 2, 3, 1)



class Zero2OneImageDomain(Domain):
    """Image domain for images in [0, 1].

    - Channel axis: 1
    - Pixel range: [0, 1]
    - Image size: arbitrary
    - Color space: RGB
    """

    def send(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return images


InternalImageDomain = Zero2OneImageDomain


class AffineDomain(Domain):
    """Image domain shifted by center and scaled by scale.

    This domain is used to convert images in [0, 1] to images in [-center, scale-center].
    In other words, the pixel intensity p in [0, 1] is converted to p * scale - center.
    """

    def __init__(
        self,
        center: np.ndarray,
        scale: float | np.ndarray,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if center.ndim == 1:  # 1D vector (C,)
            center = center[np.newaxis, :, np.newaxis, np.newaxis]
        elif center.ndim == 3:  # 3D vector (1, C, W, H)
            center = center[np.newaxis]
        else:
            raise ValueError(
                f"center must be 1D or 3D vector, but got {center.ndim}D vector."
            )
        if isinstance(scale, (float, int)) or scale.ndim == 0:
            scale = np.array([scale])[np.newaxis, np.newaxis, np.newaxis]
        elif scale.ndim == 1:  # 1D vector (C,)
            scale = scale[np.newaxis, :, np.newaxis, np.newaxis]
        elif scale.ndim == 3:  # 3D vector (1, C, W, H)
            scale = scale[np.newaxis]
        else:
            raise ValueError(
                f"scale must be scalar or 1D or 3D vector, but got {scale.ndim}D vector."
            )

        self._center = torch.from_numpy(center).to(device=device, dtype=dtype)
        self._scale = torch.from_numpy(scale).to(device=device, dtype=dtype)

    def send(self, images: torch.Tensor) -> torch.Tensor:
        return (images + self._center)  / self._scale

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return images * self._scale - self._center


class BGRDomain(Domain):
    """Image domain for BGR images."""

    def send(self, images: torch.Tensor) -> torch.Tensor:
        return _bgr2rgb(images)

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return _rgb2bgr(images)


class PILDomainWithExplicitCrop(IrreversibleDomain):
    """Image domain for PIL images.

    - Channel axis: 3
    - Pixel range: [0, 255]
    - Image size: arbitrary
    - Color space: RGB
    """

    def send(self, images: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            "PILDomainWithExplicitCrop is an irreversible domain. " \
            "It does not guarantee the reversibility of `send` and `receive` " \
            "methods. Please use PILDomainWithExplicitCrop.send() with caution.",
            RuntimeWarning,
        )
        return _to_channel_first(images) / 255.0  # to [0, 1.0]

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        images = _to_channel_last(images) * 255.0

        # Crop values to [0, 255]
        return torch.clamp(images, 0, 255)


class BdPyVGGDomain(ComposedDomain):
    """Image domain for VGG architecture defined in BdPy.

    - Channel axis: 1
    - Pixel range:
        - red: [-123, 132]
        - green: [-117, 138]
        - blue: [-104, 151]
        # These values are calculated from the mean vector of ImageNet ([123, 117, 104]).
    - Image size: arbitrary
    - Color space: BGR
    """

    def __init__(
        self, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(
            [
                AffineDomain(
                    center=np.array([123.0, 117.0, 104.0]),
                    scale=255.0,
                    device=device,
                    dtype=dtype,
                ),
                BGRDomain(),
            ]
        )
