"""Benign image transforms with deterministic, physical-unit magnitude grids.

Every transform is a pure NumPy function of a batch of float images with
values in ``[0, 1]``, laid out as ``(N, H, W)`` or ``(N, H, W, C)``.  Each
transform is parameterised by a scalar (or, for shift, a pixel pair) whose
*distance* from the identity parameter is the quantity ProbeShift measures:

==================  ======================  ==========================
transform           parameter               distance (unit)
==================  ======================  ==========================
``brightness``      added intensity delta   ``|delta|`` (intensity)
``contrast``        multiplicative factor   ``|factor - 1|``
``blur``            Gaussian sigma          sigma (pixels)
``noise``           Gaussian noise sigma    sigma (intensity)
``shift``           ``(dx, dy)`` pixels     ``max(|dx|, |dy|)`` (px)
``rotation``        angle                   ``|angle|`` (degrees)
==================  ======================  ==========================

Grids are fixed constants, ordered by non-decreasing distance (positive
side first on ties), so a probe run is fully determined by its inputs and
its recorded config.  There is no optimisation anywhere in this module:
magnitudes are swept, never searched adversarially.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

Param = Union[float, Tuple[int, int]]


# ---------------------------------------------------------------------------
# Primitive operations (batch layout: (N, H, W) or (N, H, W, C))
# ---------------------------------------------------------------------------

def _add_brightness(batch: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(batch + delta, 0.0, 1.0)


def _apply_contrast(batch: np.ndarray, factor: float) -> np.ndarray:
    mean = batch.mean(axis=(1, 2), keepdims=True)
    return np.clip(mean + (batch - mean) * factor, 0.0, 1.0)


def _gaussian_kernel(sigma: float) -> Tuple[np.ndarray, int]:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum(), radius


def _convolve_axis_edge(batch: np.ndarray, kernel: np.ndarray, radius: int,
                        axis: int) -> np.ndarray:
    pad = [(0, 0)] * batch.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(batch, pad, mode="edge")
    out = np.zeros_like(batch)
    for i, w in enumerate(kernel):
        window = [slice(None)] * batch.ndim
        window[axis] = slice(i, i + batch.shape[axis])
        out += w * padded[tuple(window)]
    return out


def _apply_blur(batch: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return batch.copy()
    kernel, radius = _gaussian_kernel(sigma)
    out = _convolve_axis_edge(batch, kernel, radius, axis=1)
    out = _convolve_axis_edge(out, kernel, radius, axis=2)
    return out


def _apply_noise(batch: np.ndarray, sigma: float,
                 field: np.ndarray) -> np.ndarray:
    if sigma == 0.0:
        return batch.copy()
    return np.clip(batch + sigma * field, 0.0, 1.0)


def _apply_rotation(batch: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0.0:
        return batch.copy()
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    height, width = batch.shape[1], batch.shape[2]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    # Inverse map: source coordinates for each output pixel.
    sx = np.clip(cx + cos_t * dx + sin_t * dy, 0.0, width - 1)
    sy = np.clip(cy - sin_t * dx + cos_t * dy, 0.0, height - 1)
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx, wy = sx - x0, sy - y0
    if batch.ndim == 4:
        top = batch[:, y0, x0, :] * (1 - wx[..., None]) \
            + batch[:, y0, x1, :] * wx[..., None]
        bottom = batch[:, y1, x0, :] * (1 - wx[..., None]) \
            + batch[:, y1, x1, :] * wx[..., None]
        return top * (1 - wy[..., None]) + bottom * wy[..., None]
    top = batch[:, y0, x0] * (1 - wx) + batch[:, y0, x1] * wx
    bottom = batch[:, y1, x0] * (1 - wx) + batch[:, y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def _apply_shift(batch: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return batch.copy()
    height, width = batch.shape[1], batch.shape[2]
    pad = [(0, 0), (abs(dy), abs(dy)), (abs(dx), abs(dx))]
    if batch.ndim == 4:
        pad.append((0, 0))
    padded = np.pad(batch, pad, mode="edge")
    y0, x0 = abs(dy) - dy, abs(dx) - dx
    if batch.ndim == 4:
        return padded[:, y0:y0 + height, x0:x0 + width, :].copy()
    return padded[:, y0:y0 + height, x0:x0 + width].copy()


# ---------------------------------------------------------------------------
# Transform definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transform:
    """One benign transform and its fixed magnitude grid.

    ``params`` are listed in evaluation order: non-decreasing distance,
    positive side before negative side on ties.  ``kind`` is ``"scalar"``
    for single-parameter transforms (refinable by bisection) and
    ``"shift"`` for the pixel-pair transform (grid resolution only).
    """

    name: str
    unit: str
    identity: float
    params: Tuple[Param, ...]
    kind: str = "scalar"

    def distance(self, param: Param) -> float:
        # Rounded to 10 decimals: grid constants carry at most 2 decimals,
        # and raw float subtraction (e.g. |0.85 - 1| vs |1.15 - 1|) would
        # otherwise make nominally equal distances differ in the last bit.
        if self.kind == "shift":
            dx, dy = param  # type: ignore[misc]
            return float(max(abs(dx), abs(dy)))
        return round(abs(float(param) - self.identity), 10)  # type: ignore[arg-type]

    @property
    def max_distance(self) -> float:
        return max(self.distance(p) for p in self.params)

    def distances(self) -> Tuple[float, ...]:
        return tuple(self.distance(p) for p in self.params)

    def apply(self, batch: np.ndarray, param: Param,
              field: Optional[np.ndarray] = None) -> np.ndarray:
        if self.name == "brightness":
            return _add_brightness(batch, float(param))  # type: ignore[arg-type]
        if self.name == "contrast":
            return _apply_contrast(batch, float(param))  # type: ignore[arg-type]
        if self.name == "blur":
            return _apply_blur(batch, float(param))  # type: ignore[arg-type]
        if self.name == "noise":
            if field is None:
                raise ValueError("the noise transform requires a fixed "
                                 "standard-normal field")
            return _apply_noise(batch, float(param), field)  # type: ignore[arg-type]
        if self.name == "rotation":
            return _apply_rotation(batch, float(param))  # type: ignore[arg-type]
        if self.name == "shift":
            dx, dy = param  # type: ignore[misc]
            return _apply_shift(batch, int(dx), int(dy))
        raise ValueError(f"unknown transform implementation: {self.name!r}")


_BRIGHTNESS_GRID: Tuple[float, ...] = (
    0.02, -0.02, 0.05, -0.05, 0.08, -0.08, 0.12, -0.12,
    0.18, -0.18, 0.25, -0.25, 0.35, -0.35, 0.5, -0.5,
)

_CONTRAST_GRID: Tuple[float, ...] = (
    0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.7, 1.3,
    0.6, 1.4, 0.5, 1.5, 0.35, 1.75, 0.2, 2.0,
)

_BLUR_GRID: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)

_NOISE_GRID: Tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5,
)

_ROTATION_GRID: Tuple[float, ...] = (
    1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0, 8.0, -8.0,
    12.0, -12.0, 18.0, -18.0, 25.0, -25.0, 35.0, -35.0, 45.0, -45.0,
)

_SHIFT_GRID: Tuple[Tuple[int, int], ...] = tuple(
    pair
    for d in (1, 2, 3, 4, 6, 8)
    for pair in ((d, 0), (-d, 0), (0, d), (0, -d))
)


def default_transforms() -> Sequence[Transform]:
    """The six built-in benign transforms, in report order."""
    return (
        Transform("brightness", "intensity", 0.0, _BRIGHTNESS_GRID),
        Transform("contrast", "|factor - 1|", 1.0, _CONTRAST_GRID),
        Transform("blur", "sigma (px)", 0.0, _BLUR_GRID),
        Transform("noise", "sigma (intensity)", 0.0, _NOISE_GRID),
        Transform("shift", "px", 0.0, _SHIFT_GRID, kind="shift"),
        Transform("rotation", "degrees", 0.0, _ROTATION_GRID),
    )


_TRANSFORMS_BY_NAME = {t.name: t for t in default_transforms()}


def get_transform(name: str) -> Transform:
    try:
        return _TRANSFORMS_BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(_TRANSFORMS_BY_NAME))
        raise ValueError(
            f"unknown transform {name!r}; known transforms: {known}"
        ) from None
