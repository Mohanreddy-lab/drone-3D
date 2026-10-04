"""Geodesy and rigid geometry.

All reconstruction work happens in a local East-North-Up (ENU) frame anchored
at the first GPS fix: metres, right-handed, z up. That keeps float32 point
clouds precise and makes measurements plain Euclidean geometry. Exports are
converted to WGS84 (lat/lon) and to the matching UTM zone at the very end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# WGS84 ellipsoid
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, h: np.ndarray) -> np.ndarray:
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    sin_lat = np.sin(lat_r)
    n = _A / np.sqrt(1.0 - _E2 * sin_lat**2)
    x = (n + h) * np.cos(lat_r) * np.cos(lon_r)
    y = (n + h) * np.cos(lat_r) * np.sin(lon_r)
    z = (n * (1.0 - _E2) + h) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bowring's method, iterated -- sub-millimetre for terrestrial heights."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _E2))
    for _ in range(5):
        n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - _E2 * n / (n + h)))
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def _enu_rotation(lat0: float, lon0: float) -> np.ndarray:
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array(
        [
            [-np.sin(lo), np.cos(lo), 0.0],
            [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
            [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
        ]
    )


@dataclass(frozen=True)
class LocalFrame:
    """ENU frame anchored at a WGS84 origin."""

    lat0: float
    lon0: float
    h0: float = 0.0

    def to_enu(self, lat: np.ndarray, lon: np.ndarray, h: np.ndarray) -> np.ndarray:
        ecef = geodetic_to_ecef(np.asarray(lat), np.asarray(lon), np.asarray(h))
        origin = geodetic_to_ecef(np.array(self.lat0), np.array(self.lon0), np.array(self.h0))
        return (ecef - origin) @ _enu_rotation(self.lat0, self.lon0).T

    def to_geodetic(self, enu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origin = geodetic_to_ecef(np.array(self.lat0), np.array(self.lon0), np.array(self.h0))
        ecef = np.asarray(enu) @ _enu_rotation(self.lat0, self.lon0) + origin
        return ecef_to_geodetic(ecef)

    def utm_epsg(self) -> int:
        zone = int((self.lon0 + 180.0) // 6.0) + 1
        return (32600 if self.lat0 >= 0 else 32700) + zone


@dataclass(frozen=True)
class Similarity:
    """x_world = scale * R @ x_model + t"""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(pts) @ self.rotation.T + self.translation

    @staticmethod
    def identity() -> Similarity:
        return Similarity(1.0, np.eye(3), np.zeros(3))


def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> Similarity:
    """Least-squares similarity transform mapping ``src`` onto ``dst``."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c * w[:, None]).T @ s_c
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rot = u @ sign @ vt
    var_s = (w * (s_c**2).sum(axis=1)).sum()
    scale = float(np.trace(np.diag(d) @ sign) / var_s)
    t = mu_d - scale * rot @ mu_s
    return Similarity(scale, rot, t)


def robust_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float,
    iterations: int = 500,
    seed: int = 0,
) -> tuple[Similarity, np.ndarray]:
    """RANSAC over minimal 3-point samples, then refit on inliers.

    GPS has outliers (multipath near buildings, the first seconds after
    take-off) and a single bad fix should not tilt the whole model.
    """
    n = len(src)
    if n < 3:
        raise ValueError("need at least 3 correspondences to georeference")
    rng = np.random.default_rng(seed)
    best_inliers = np.ones(n, bool)
    best_count = -1
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        # Skip near-collinear samples: they leave rotation about the line free.
        a, b, c = src[idx]
        if np.linalg.norm(np.cross(b - a, c - a)) < 1e-9:
            continue
        sim = umeyama(src[idx], dst[idx])
        err = np.linalg.norm(sim.apply(src) - dst, axis=1)
        inliers = err < threshold
        if inliers.sum() > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers
    if best_count < 3:
        best_inliers = np.ones(n, bool)
    sim = umeyama(src[best_inliers], dst[best_inliers])
    # One refinement pass with the threshold applied to the refit.
    err = np.linalg.norm(sim.apply(src) - dst, axis=1)
    refined = err < threshold
    if refined.sum() >= 3:
        best_inliers = refined
        sim = umeyama(src[best_inliers], dst[best_inliers])
    return sim, best_inliers


def look_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World(ENU)-from-camera rotation for a gimbal camera.

    Camera axes follow the OpenCV/COLMAP convention: x right, y down,
    z forward. ``yaw`` is heading clockwise from north, ``pitch`` is gimbal
    pitch with -90 meaning straight down.
    """
    psi, th = np.radians(yaw_deg), np.radians(pitch_deg)
    horiz = np.array([np.sin(psi), np.cos(psi), 0.0])
    fwd = np.cos(th) * horiz + np.sin(th) * np.array([0.0, 0.0, 1.0])
    right = np.array([np.cos(psi), -np.sin(psi), 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)
