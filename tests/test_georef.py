import numpy as np
import pytest

from drone3d.geo import Similarity, look_rotation
from drone3d.georef import _rotation_between, estimate_down, level


def _tilt(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _survey(n=40, seed=0, turns=True):
    rng = np.random.default_rng(seed)
    yaws = np.where(np.arange(n) % 20 < 10, 0.0, 180.0) + rng.normal(0, 3, n)
    if turns:  # U-turns between survey lines sweep through other headings
        yaws[::5] = rng.uniform(0, 360, len(yaws[::5]))
    rots = [look_rotation(y, -75.0) for y in yaws]
    centers = np.column_stack([rng.uniform(-30, 30, n), rng.uniform(-40, 40, n), np.full(n, 40.0)])
    return rots, centers


def test_rotation_between():
    a, b = np.array([0.1, 0.2, -1.0]), np.array([0.0, 0.0, -1.0])
    R = _rotation_between(a, b)
    assert np.allclose(R @ (a / np.linalg.norm(a)), b)


@pytest.mark.parametrize("use_pitch", [True, False])
def test_gimbal_recovers_tilt(use_pitch):
    rots, _ = _survey()
    T = _tilt(3.0)  # model frame tilted 3 degrees
    model_rots = [T @ R for R in rots]
    g, method = estimate_down(model_rots, np.full(len(rots), -75.0) if use_pitch else None)
    assert np.allclose(g, T @ np.array([0, 0, -1.0]), atol=1e-6)
    assert method == ("gimbal-pitch" if use_pitch else "level-horizon")


def test_parallel_headings_without_pitch_are_unobservable():
    rots, _ = _survey(turns=False)
    rots = [look_rotation(y, -75.0) for y in [0.0, 180.0] * 10]
    assert estimate_down(rots, None) is None


def test_level_removes_tilt_that_gps_heights_introduce():
    rots, true_c = _survey()
    T = _tilt(2.5)
    # Model frame == world, but the GPS-only fit came out tilted by T.
    tilted = Similarity(1.0, T, np.zeros(3))
    pitch = np.full(len(rots), -75.0)
    fixed, info = level(tilted, true_c, true_c, rots, pitch, np.ones(len(rots), bool))
    assert info["tilt_correction_deg"] == pytest.approx(2.5, abs=0.01)
    assert np.allclose(fixed.apply(true_c), true_c, atol=1e-6)
