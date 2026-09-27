"""Spatial grid and small vectorised geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Grid:
    """Regular 2-D grid in the dataset frame (metres, x right, y down)."""

    x0: float
    y0: float
    nx: int
    ny: int
    cell: float

    @classmethod
    def around(cls, x: np.ndarray, y: np.ndarray, cell: float = 1.0, pad: float = 2.0) -> "Grid":
        x0, y0 = np.floor(x.min() - pad), np.floor(y.min() - pad)
        nx = int(np.ceil((x.max() + pad - x0) / cell))
        ny = int(np.ceil((y.max() + pad - y0) / cell))
        return cls(float(x0), float(y0), nx, ny, cell)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """matplotlib imshow extent with y pointing down."""
        return (self.x0, self.x0 + self.nx * self.cell, self.y0 + self.ny * self.cell, self.y0)

    def ij(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((np.asarray(x) - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.y0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy, ix, ok

    def flat(self, x, y) -> np.ndarray:
        """Flat cell index; -1 outside the grid."""
        iy, ix, ok = self.ij(x, y)
        return np.where(ok, iy * self.nx + ix, -1)

    def accumulate(self, x, y, weights=None) -> np.ndarray:
        idx = self.flat(x, y)
        w = np.ones(len(idx)) if weights is None else np.asarray(weights, float)
        keep = idx >= 0
        return np.bincount(idx[keep], weights=w[keep], minlength=self.size).reshape(self.shape)

    def smooth(self, field: np.ndarray, sigma_m: float) -> np.ndarray:
        return gaussian_filter(field.astype(float), sigma=sigma_m / self.cell, mode="constant")


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def segment_hits_boxes(p0, p1, centers, headings, lengths, widths) -> np.ndarray:
    """Does segment p0->p1 intersect oriented rectangles? Broadcasts over rows.

    p0, p1, centers: (..., 2); headings/lengths/widths: (...). Slab test in each
    box's local frame (Liang-Barsky clipping). Returns bool array (...).
    """
    c, s = np.cos(headings), np.sin(headings)

    def to_local(p):
        d = p - centers
        return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], axis=-1)

    a, b = to_local(p0), to_local(p1)
    d = b - a
    half = np.stack([lengths / 2, widths / 2], axis=-1)
    t_enter = np.zeros(a.shape[:-1])
    t_exit = np.ones(a.shape[:-1])
    for k in range(2):
        dk, ak, hk = d[..., k], a[..., k], half[..., k]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-hk - ak) / dk
            t2 = (hk - ak) / dk
        lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
        parallel = np.abs(dk) < 1e-12
        inside = np.abs(ak) <= hk
        lo = np.where(parallel, np.where(inside, -np.inf, np.inf), lo)
        hi = np.where(parallel, np.where(inside, np.inf, -np.inf), hi)
        t_enter = np.maximum(t_enter, lo)
        t_exit = np.minimum(t_exit, hi)
    return t_enter <= t_exit
