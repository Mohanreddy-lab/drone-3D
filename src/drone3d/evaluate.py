"""Score a reconstruction of the synthetic site against its ground truth."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from drone3d import export, measure
from drone3d.geo import LocalFrame
from drone3d.synth import Box, Pyramid, Site


def evaluate(run_dir: str | Path, truth_path: str | Path, gsd: float = 0.2) -> dict:
    run = Path(run_dir)
    truth = json.loads(Path(truth_path).read_text())
    report = json.loads((run / "report.json").read_text())
    o = report["origin"]
    run_frame = LocalFrame(o["lat"], o["lon"], o["h"])
    t = truth["origin"]
    truth_frame = LocalFrame(t["lat"], t["lon"], t["h"])
    site = Site(
        boxes=[Box(**b) for b in truth["boxes"]],
        pyramids=[
            Pyramid(**{k: v for k, v in p.items() if k != "volume"}) for p in truth["pyramids"]
        ],
    )

    def to_truth(p: np.ndarray) -> np.ndarray:
        lat, lon, h = run_frame.to_geodetic(p.astype(np.float64))
        return truth_frame.to_enu(lat, lon, h)

    pts, rgb = export.read_ply(run / "cloud.ply")
    P_abs = to_truth(pts)
    # Separate *absolute* error (where the model sits on the globe, limited by
    # GPS) from *relative* error (shape and size, what measurements depend
    # on): find the rigid shift that best lays the model onto the truth, then
    # score shape and measurements after removing it -- just as a surveyor
    # draws a stockpile outline around the pile they see in the model.
    shift = _best_shift(P_abs, site)
    P = P_abs + shift

    # Surface error: vertical distance to the true heightfield, away from walls
    # (a point on a wall has no single "true height").
    near_edge = np.zeros(len(P), bool)
    for b in site.boxes:
        for x in (b.x0, b.x1):
            near_edge |= (
                (np.abs(P[:, 0] - x) < 0.6) & (P[:, 1] > b.y0 - 0.6) & (P[:, 1] < b.y1 + 0.6)
            )
        for y in (b.y0, b.y1):
            near_edge |= (
                (np.abs(P[:, 1] - y) < 0.6) & (P[:, 0] > b.x0 - 0.6) & (P[:, 0] < b.x1 + 0.6)
            )
    surf = ~near_edge
    dz = P[surf, 2] - site.height_at(P[surf, 0], P[surf, 1])
    surface = {
        "model_offset_m": (-shift).round(3).tolist(),
        "points_scored": int(surf.sum()),
        "vertical_bias_m": float(np.median(dz)),
        "vertical_rmse_m": float(np.sqrt(np.mean(dz**2))),
        "vertical_mae_m": float(np.median(np.abs(dz - np.median(dz)))),
        "within_25cm_percent": float(100 * np.mean(np.abs(dz - np.median(dz)) < 0.25)),
    }

    # Camera positions: solved centres vs the true flight path.
    cams = json.loads((run / "scene.json").read_text())["cameras"]
    fps = truth["fps"]
    true_pos = np.array(truth["true_positions"])
    errs = []
    for c in cams:
        idx = int(c["name"].split("_")[1].split(".")[0])
        if idx < len(true_pos):
            errs.append(to_truth(np.array([c["center"]]))[0] - true_pos[idx])
    errs = np.array(errs)
    camera = {
        "cameras": len(errs),
        "abs_h_rmse_m": float(np.sqrt(np.mean(np.sum(errs[:, :2] ** 2, 1)))),
        "abs_v_rmse_m": float(np.sqrt(np.mean(errs[:, 2] ** 2))),
        "mean_offset_m": errs.mean(axis=0).round(3).tolist(),
        "fps": fps,
    }

    # Measurements a surveyor would actually take.
    dsm, _ortho, x0, y_top = export.rasterize(P.astype(np.float32), rgb, gsd)
    grid = measure.Grid(dsm, x0, y_top, gsd)
    objects = []
    for b in site.boxes:
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        roof = _cells(grid, b.x0 + 1.5, b.x1 - 1.5, b.y0 + 1.5, b.y1 - 1.5)
        ring = _ring(grid, b, 3.0, 8.0)
        h = float(np.nanmedian(roof) - np.nanmedian(ring))
        # Footprint from the DSM: extent of cells above half height, along
        # the centre lines of the building.
        half = np.nanmedian(ring) + h / 2
        ex = _extent(grid, cy, (b.x0 - 6, b.x1 + 6), half, axis="x")
        ey = _extent(grid, cx, (b.y0 - 6, b.y1 + 6), half, axis="y")
        objects.append(
            {
                "name": b.name,
                "true": {
                    "length_x_m": b.x1 - b.x0,
                    "length_y_m": b.y1 - b.y0,
                    "height_m": b.height,
                },
                "measured": {"length_x_m": ex, "length_y_m": ey, "height_m": h},
            }
        )
    for p in site.pyramids:
        s = p.base / 2 + 2.0
        poly = [
            (p.cx - s, p.cy - s),
            (p.cx + s, p.cy - s),
            (p.cx + s, p.cy + s),
            (p.cx - s, p.cy + s),
        ]
        v = measure.volume(grid, poly, base="plane")
        objects.append(
            {
                "name": p.name,
                "true": {"volume_m3": p.volume, "height_m": p.height},
                "measured": {"volume_m3": v["cut_m3"], "coverage": v["coverage"]},
            }
        )
    result = {"surface": surface, "camera_positions": camera, "objects": objects}
    (run / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


def _best_shift(P: np.ndarray, site: Site, span: float = 8.0) -> np.ndarray:
    """Translation aligning model points to the true heightfield.

    Coarse-to-fine grid search on horizontal shift; the vertical shift is the
    median residual at each candidate. Scored by median absolute residual,
    which ignores the wall points a heightfield cannot represent.
    """
    rng = np.random.default_rng(0)
    sub = P[rng.choice(len(P), min(len(P), 30000), replace=False)]
    best = (np.inf, np.zeros(3))
    center = np.zeros(2)
    for step, half in ((0.5, span), (0.1, 0.6), (0.02, 0.12)):
        offs = np.arange(-half, half + 1e-9, step)
        for dx in offs:
            for dy in offs:
                x = sub[:, 0] + center[0] + dx
                y = sub[:, 1] + center[1] + dy
                r = sub[:, 2] - site.height_at(x, y)
                dz = -float(np.median(r))
                score = float(np.median(np.abs(r + dz)))
                if score < best[0]:
                    best = (score, np.array([center[0] + dx, center[1] + dy, dz]))
        center = best[1][:2].copy()
    return best[1]


def _cells(grid: measure.Grid, x0, x1, y0, y1) -> np.ndarray:
    xs = np.arange(x0, x1, grid.gsd)
    ys = np.arange(y0, y1, grid.gsd)
    X, Y = np.meshgrid(xs, ys)
    return grid.sample(np.column_stack([X.ravel(), Y.ravel()]))


def _ring(grid: measure.Grid, b: Box, inner: float, outer: float) -> np.ndarray:
    z = _cells(grid, b.x0 - outer, b.x1 + outer, b.y0 - outer, b.y1 + outer)
    xs = np.arange(b.x0 - outer, b.x1 + outer, grid.gsd)
    ys = np.arange(b.y0 - outer, b.y1 + outer, grid.gsd)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    inside = (b.x0 - inner < X) & (b.x1 + inner > X) & (b.y0 - inner < Y) & (b.y1 + inner > Y)
    return z[~inside]


def _extent(grid: measure.Grid, at: float, span, threshold: float, axis: str) -> float:
    s = np.arange(span[0], span[1], grid.gsd / 2)
    # Median over a 3 m wide band to be robust to single noisy rows.
    profiles = []
    for off in np.linspace(-1.5, 1.5, 7):
        xy = (
            np.column_stack([s, np.full_like(s, at + off)])
            if axis == "x"
            else np.column_stack([np.full_like(s, at + off), s])
        )
        profiles.append(grid.sample(xy))
    with warnings.catch_warnings():  # columns outside the model are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        z = np.nanmedian(np.array(profiles), axis=0)
    above = np.nonzero(z > threshold)[0]
    if len(above) == 0:
        return float("nan")
    return float(s[above[-1]] - s[above[0]] + grid.gsd / 2)
