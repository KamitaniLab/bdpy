"""Image domains for PyTorch.

This module provides image domains for PyTorch. The image domains are used to
convert images between each domain and library's internal common space.
The internal common space is defined as follows:

- Channel axis: 1
- Pixel range: [0, 1]
- Image size: arbitrary
- Color space: RGB
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torchvision.transforms import InterpolationMode, Resize

from .core import Domain, InternalDomain, IrreversibleDomain, ComposedDomain


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


# NOTE: The internal common space for images is defined as follows:
# - Channel axis: 1
# - Pixel range: [0, 1]
# - Image size: arbitrary
# - Color space: RGB
Zero2OneImageDomain = InternalDomain[torch.Tensor]


class AffineDomain(Domain):
    """Image domain shifted by center and scaled by scale.

    This domain is used to convert images in [0, 1] to images in [-center, scale-center].
    In other words, the pixel intensity p in [0, 1] is converted to p * scale - center.

    Parameters
    ----------
    center : float | np.ndarray
        Center of the affine transformation.
        If center.ndim == 0, it must be scalar.
        If center.ndim == 1, it must be 1D vector (C,).
        If center.ndim == 3, it must be 3D vector (1, C, W, H).
    scale : float | np.ndarray
        Scale of the affine transformation.
        If scale.ndim == 0, it must be scalar.
        If scale.ndim == 1, it must be 1D vector (C,).
        If scale.ndim == 3, it must be 3D vector (1, C, W, H).
    device : torch.device | None
        Device to send/receive images.
    dtype : torch.dtype | None
        Data type to send/receive images.
    """

    def __init__(
        self,
        center: float | np.ndarray,
        scale: float | np.ndarray,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if isinstance(center, (float, int)) or center.ndim == 0:
            center = np.array([center])[np.newaxis, np.newaxis, np.newaxis]
        elif center.ndim == 1:  # 1D vector (C,)
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
        return _to_channel_first(images) / 255.0  # to [0, 1.0]

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            "`PILDominWithExplicitCrop.receive` performs explicit cropping. " \
            "It could be affected to the gradient computation. " \
            "Please do not use this domain inside the optimization.",
            RuntimeWarning,
        )

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
    - Image size: arbitrary
    - Color space: BGR

    Parameters
    ----------
    device : torch.device | None
        Device to send/receive images.
    dtype : torch.dtype | None
        Data type to send/receive images.

    Notes
    -----
    The pixel ranges of this domain are derived from the mean vector of ImageNet ([123, 117, 104]).
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


class FixedResolutionDomain(IrreversibleDomain):
    """Image domain for images with fixed resolution.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Spatial resolution of the images.
    interpolation : InterpolationMode, optional
        Interpolation mode for resizing. (default: InterpolationMode.BILINEAR)
    antialias : bool, optional
        Whether to use antialiasing. (default: True)
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self._image_shape = image_shape
        self._interpolation = interpolation
        self._antialias = antialias

        self._resizer = Resize(
            size=self._image_shape,
            interpolation=self._interpolation,
            antialias=self._antialias
        )

    def send(self, images: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "FixedResolutionDomain is not supposed to be used for sending images " \
            "because the internal image resolution could not be determined."
        )

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return self._resizer(images)
