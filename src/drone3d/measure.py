"""Measurements on the reconstructed surface.

The web viewer implements the same formulas in JavaScript for interactive
use; this module is the reference implementation used by the CLI, the
evaluation against ground truth, and the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Grid:
    """A north-up raster in local ENU metres (row 0 = northern edge)."""

    z: np.ndarray
    x0: float
    y_top: float
    gsd: float

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.stack([(xy[:, 0] - self.x0) / self.gsd, (self.y_top - xy[:, 1]) / self.gsd], 1)

    def sample(self, xy: np.ndarray) -> np.ndarray:
        px = np.round(self.to_pixel(xy)).astype(int)
        h, w = self.z.shape
        px[:, 0] = np.clip(px[:, 0], 0, w - 1)
        px[:, 1] = np.clip(px[:, 1], 0, h - 1)
        return self.z[px[:, 1], px[:, 0]]


def distance(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    return {
        "slope_m": float(np.linalg.norm(d)),
        "horizontal_m": float(np.linalg.norm(d[:2])),
        "vertical_m": float(d[2]),
    }


def polygon_area(xy) -> float:
    """Planimetric (map) area of a polygon, shoelace formula."""
    p = np.asarray(xy, float)[:, :2]
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _polygon_mask(grid: Grid, polygon) -> np.ndarray:
    px = grid.to_pixel(np.asarray(polygon, float)[:, :2])
    mask = np.zeros(grid.z.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(px * 8).astype(np.int32)], 1, shift=3)
    return mask.astype(bool)


def volume(grid: Grid, polygon, base: str | float = "plane") -> dict:
    """Cut/fill volume inside ``polygon`` relative to a base surface.

    ``base="plane"`` fits a plane to the surface along the polygon outline --
    the standard stockpile method, which follows sloping ground. ``"min"``
    uses the lowest boundary height; a number sets an absolute base height.
    """
    poly = np.asarray(polygon, float)[:, :2]
    mask = _polygon_mask(grid, poly)
    # Sample the outline densely for the base surface.
    edge = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        n = max(int(np.linalg.norm(b - a) / grid.gsd), 2)
        edge.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    edge = np.concatenate(edge)
    ez = grid.sample(edge)
    ok = np.isfinite(ez)
    edge, ez = edge[ok], ez[ok]
    rows, cols = np.nonzero(mask)
    xs = grid.x0 + cols * grid.gsd
    ys = grid.y_top - rows * grid.gsd
    if isinstance(base, (int, float)):
        bz = np.full(len(xs), float(base))
        base_desc = f"fixed {base:.2f} m"
    elif base == "min":
        bz = np.full(len(xs), float(np.min(ez)))
        base_desc = "lowest boundary point"
    else:
        A = np.column_stack([edge, np.ones(len(edge))])
        coef, *_ = np.linalg.lstsq(A, ez, rcond=None)
        bz = coef[0] * xs + coef[1] * ys + coef[2]
        base_desc = "plane fitted to boundary"
    z = grid.z[rows, cols]
    valid = np.isfinite(z)
    dz = z[valid] - bz[valid]
    cell = grid.gsd**2
    return {
        "cut_m3": float(np.sum(np.clip(dz, 0, None)) * cell),
        "fill_m3": float(np.sum(np.clip(-dz, 0, None)) * cell),
        "area_m2": float(mask.sum() * cell),
        "coverage": float(valid.mean()) if len(valid) else 0.0,
        "base": base_desc,
    }
