import numpy as np
import pytest

from drone3d.geo import LocalFrame, Similarity, look_rotation, robust_similarity, umeyama


def test_enu_roundtrip_is_submillimetre():
    f = LocalFrame(17.385, 78.4867, 520.0)
    enu = np.array([[0, 0, 0], [120.5, -80.25, 35.0], [-1500, 2200, -12]], float)
    lat, lon, h = f.to_geodetic(enu)
    back = f.to_enu(lat, lon, h)
    assert np.abs(back - enu).max() < 1e-3


def test_enu_axes_point_east_north_up():
    f = LocalFrame(17.385, 78.4867, 0.0)
    east = f.to_enu(np.array(17.385), np.array(78.4877), np.array(0.0))
    north = f.to_enu(np.array(17.386), np.array(78.4867), np.array(0.0))
    assert east[0] > 100 and abs(east[1]) < 0.1
    assert north[1] > 100 and abs(north[0]) < 0.1


def test_utm_zone():
    assert LocalFrame(17.4, 78.5).utm_epsg() == 32644  # Hyderabad: zone 44N
    assert LocalFrame(-33.9, 18.4).utm_epsg() == 32734  # Cape Town: zone 34S


def _random_sim(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return Similarity(2.7, q, np.array([10.0, -4.0, 3.0]))


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    sim = _random_sim(rng)
    src = rng.normal(size=(30, 3)) * 10
    est = umeyama(src, sim.apply(src))
    assert est.scale == pytest.approx(2.7, rel=1e-9)
    assert np.allclose(est.rotation, sim.rotation)
    assert np.allclose(est.translation, sim.translation)


def test_robust_similarity_ignores_gps_outliers():
    rng = np.random.default_rng(2)
    sim = _random_sim(rng)
    src = rng.normal(size=(60, 3)) * 10
    dst = sim.apply(src) + rng.normal(scale=0.3, size=(60, 3))
    dst[:6] += 50.0  # multipath spikes
    est, inl = robust_similarity(src, dst, threshold=2.0)
    assert not inl[:6].any() and inl[6:].mean() > 0.9
    assert est.scale == pytest.approx(2.7, rel=0.01)


def test_look_rotation_nadir_and_level():
    nadir = look_rotation(0.0, -90.0)
    assert np.allclose(nadir[:, 2], [0, 0, -1], atol=1e-12)  # forward = down
    level = look_rotation(90.0, 0.0)  # heading east
    assert np.allclose(level[:, 2], [1, 0, 0], atol=1e-12)
    assert np.allclose(level[:, 1], [0, 0, -1], atol=1e-12)  # image down = world down
    assert np.linalg.det(level) == pytest.approx(1.0)
