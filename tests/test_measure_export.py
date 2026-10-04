import numpy as np
import pytest

from drone3d import export, measure
from drone3d.synth import Site


def _site_grid(gsd=0.1):
    site = Site()
    xs = np.arange(-60, 60, gsd)
    ys = np.arange(-60, 60, gsd)
    X, Y = np.meshgrid(xs, ys[::-1])
    Z = site.height_at(X, Y).astype(np.float32)
    return site, measure.Grid(Z, float(xs[0]), float(ys[-1]), gsd)


def test_pyramid_volume():
    site, grid = _site_grid()
    p = site.pyramids[0]
    s = p.base / 2 + 2
    poly = [(p.cx - s, p.cy - s), (p.cx + s, p.cy - s), (p.cx + s, p.cy + s), (p.cx - s, p.cy + s)]
    v = measure.volume(grid, poly)
    assert v["cut_m3"] == pytest.approx(512.0, rel=0.02)
    assert v["fill_m3"] < 1.0


def test_volume_on_sloping_ground_uses_fitted_plane():
    site, grid = _site_grid()
    X0 = grid.x0 + np.arange(grid.z.shape[1]) * grid.gsd
    tilt = (0.05 * X0)[None, :].astype(np.float32)  # 5 % slope
    g2 = measure.Grid(grid.z + tilt, grid.x0, grid.y_top, grid.gsd)
    p = site.pyramids[0]
    s = p.base / 2 + 2
    poly = [(p.cx - s, p.cy - s), (p.cx + s, p.cy - s), (p.cx + s, p.cy + s), (p.cx - s, p.cy + s)]
    assert measure.volume(g2, poly)["cut_m3"] == pytest.approx(512.0, rel=0.02)


def test_area_and_distance():
    assert measure.polygon_area([(0, 0), (20, 0), (20, 12), (0, 12)]) == pytest.approx(240.0)
    d = measure.distance((0, 0, 0), (3, 4, 12))
    assert d["slope_m"] == pytest.approx(13.0) and d["horizontal_m"] == pytest.approx(5.0)


def test_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)).astype(np.float32)
    rgb = rng.integers(0, 255, (500, 3)).astype(np.uint8)
    export.write_ply(tmp_path / "c.ply", pts, rgb)
    p2, c2 = export.read_ply(tmp_path / "c.ply")
    assert np.array_equal(pts, p2) and np.array_equal(rgb, c2)


def test_rasterize_keeps_highest_point():
    pts = np.array([[0.05, 0.05, 1.0], [0.06, 0.06, 5.0], [1.05, 0.05, 2.0]], np.float32)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    dsm, ortho, x0, y_top = export.rasterize(pts, rgb, 0.5)
    assert np.nanmax(dsm) == pytest.approx(5.0)
