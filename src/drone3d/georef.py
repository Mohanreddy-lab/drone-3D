"""Georeferencing and accuracy estimation.

The sparse model comes out of SfM in an arbitrary frame (or, with GPS-prior
bundle adjustment, roughly in ENU already). We fit a robust 7-parameter
similarity from SfM camera centres to GPS positions, optionally search for the
video/telemetry clock offset that best explains the fit, and bootstrap the fit
to put an honest error bar on every length the viewer will measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pycolmap

from drone3d.frames import SelectedFrame
from drone3d.geo import Similarity, robust_similarity, umeyama
from drone3d.sfm import camera_centers
from drone3d.telemetry import Telemetry


@dataclass
class GeorefResult:
    similarity: Similarity
    time_offset: float
    inliers: int
    total: int
    residuals: np.ndarray  # per-inlier camera, metres (ENU)
    scale_sigma_rel: float
    stats: dict = field(default_factory=dict)


def _img_id(rec: pycolmap.Reconstruction, name: str) -> int:
    img = rec.find_image_with_name(name)
    return img.image_id


def _fit(model_c: np.ndarray, gps: np.ndarray, threshold: float):
    sim, inl = robust_similarity(model_c, gps, threshold)
    res = sim.apply(model_c[inl]) - gps[inl]
    return sim, inl, res


def georeference(
    rec: pycolmap.Reconstruction,
    frames: list[SelectedFrame],
    telemetry: Telemetry | None,
    frame,
    *,
    threshold: float = 6.0,
    search_offset: float = 1.0,
    initial_offset: float = 0.0,
    use_gravity: bool = True,
    log=print,
) -> GeorefResult:
    centers = camera_centers(rec)
    by_name = {f.name: f for f in frames}
    names = [n for n in centers if by_name.get(n) and by_name[n].enu is not None]
    if len(names) < 3:
        raise RuntimeError("fewer than 3 registered frames have GPS; cannot georeference")
    model_c = np.array([centers[n] for n in names])
    times = np.array([by_name[n].time for n in names])

    def gps_at(offset: float) -> np.ndarray:
        if telemetry is None:
            return np.array([by_name[n].enu for n in names])
        return telemetry.positions_at(times, frame, offset)

    best = None
    offsets = [initial_offset]
    if telemetry is not None and search_offset > 0:
        coarse = np.arange(-search_offset, search_offset + 1e-9, 0.1) + initial_offset
        offsets = list(coarse)
    for off in offsets:
        sim, inl, res = _fit(model_c, gps_at(off), threshold)
        rms = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))
        score = (int(inl.sum()), -rms)
        if best is None or score > best[0]:
            best = (score, off)
    assert best is not None
    off = best[1]
    if telemetry is not None and search_offset > 0:  # refine around the coarse best
        fine = np.arange(off - 0.1, off + 0.1001, 0.02)
        cands = []
        for o in fine:
            sim, inl, res = _fit(model_c, gps_at(o), threshold)
            cands.append(((int(inl.sum()), -float(np.sqrt(np.mean(np.sum(res**2, 1))))), o))
        off = max(cands)[1]
    gps = gps_at(off)
    sim, inl, res = _fit(model_c, gps, threshold)
    level_stats: dict = {"levelling": "disabled"}
    if use_gravity:
        rots = [
            np.asarray(rec.images[_img_id(rec, n)].cam_from_world().rotation.matrix()).T
            for n in names
        ]
        pitch = None
        if telemetry is not None and telemetry.gimbal_pitch is not None:
            pitch = telemetry.interp_optional(telemetry.gimbal_pitch, times + off)
        sim, level_stats = level(sim, model_c, gps, rots, pitch, inl)
        res = sim.apply(model_c[inl]) - gps[inl]

    # Bootstrap: how well do the GPS fixes pin down scale?
    rng = np.random.default_rng(0)
    idx_in = np.nonzero(inl)[0]
    scales = []
    for _ in range(200):
        sample = rng.choice(idx_in, len(idx_in), replace=True)
        if len(np.unique(sample)) < 3:
            continue
        scales.append(umeyama(model_c[sample], gps[sample]).scale)
    scale_sigma_rel = float(np.std(scales) / sim.scale) if scales else float("nan")

    res_h = np.linalg.norm(res[:, :2], axis=1)
    res_v = np.abs(res[:, 2])
    stats = {
        "cameras_used": int(inl.sum()),
        "cameras_with_gps": len(names),
        "time_offset_s": round(float(off), 3),
        "gps_residual_rms_h_m": float(np.sqrt(np.mean(res_h**2))),
        "gps_residual_rms_v_m": float(np.sqrt(np.mean(res_v**2))),
        "scale_sigma_percent": 100 * scale_sigma_rel,
        "scale": sim.scale,
        **level_stats,
    }
    log(
        f"georeferenced on {stats['cameras_used']}/{len(names)} cameras; "
        f"GPS residual RMS {stats['gps_residual_rms_h_m']:.2f} m (h) "
        f"{stats['gps_residual_rms_v_m']:.2f} m (v); clock offset {off:+.2f} s; "
        f"scale uncertainty {stats['scale_sigma_percent']:.2f}%; "
        f"levelling: {stats['levelling']}"
        + (f" ({stats['tilt_correction_deg']:.2f} deg)" if "tilt_correction_deg" in stats else "")
    )
    return GeorefResult(sim, float(off), int(inl.sum()), len(names), res, scale_sigma_rel, stats)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def estimate_down(
    rotations: list[np.ndarray], pitches: np.ndarray | None
) -> tuple[np.ndarray, str] | None:
    """World "down" as seen by a stabilised gimbal, in model coordinates.

    A gimbal holds the horizon level (roll ~ 0) and logs its pitch, so each
    camera tells us where gravity points in its own frame. Averaged over all
    cameras that pins the model's tilt to a fraction of a degree, far better
    than consumer-GPS heights can over a site only tens of metres wide.

    ``rotations`` are world-from-camera matrices in the current model frame.
    With logged pitch, down is predicted per camera; without it, roll ~ 0
    alone says each camera's x-axis is horizontal, and down is the direction
    most perpendicular to all of them (needs some heading variety).
    """
    if not rotations:
        return None
    if pitches is not None and np.all(np.isfinite(pitches)):
        from drone3d.geo import look_rotation

        votes = []
        for W, pitch in zip(rotations, pitches, strict=True):
            d_cam = look_rotation(0.0, float(pitch)).T @ np.array([0.0, 0.0, -1.0])
            votes.append(W @ d_cam)
        v = np.array(votes)
        g = np.median(v, axis=0)
        return g / np.linalg.norm(g), "gimbal-pitch"
    r = np.array([W[:, 0] for W in rotations])
    evals, evecs = np.linalg.eigh(r.T @ r)
    if evals[1] < 0.05 * evals[2]:  # all headings parallel: tilt about them unobservable
        return None
    g = evecs[:, 0]
    if g[2] > 0:
        g = -g
    return g, "level-horizon"


def level(
    sim: Similarity,
    model_c: np.ndarray,
    gps: np.ndarray,
    rotations_model: list[np.ndarray],
    pitches: np.ndarray | None,
    inliers: np.ndarray,
    max_correction_deg: float = 8.0,
) -> tuple[Similarity, dict]:
    """Re-level a GPS similarity using gimbal gravity, then refit 4-DoF + scale."""
    world_rots = [sim.rotation @ R for R in rotations_model]
    est = estimate_down(world_rots, pitches)
    if est is None:
        return sim, {"levelling": "none (no attitude information)"}
    g, method = est
    Q = _rotation_between(g, np.array([0.0, 0.0, -1.0]))
    angle = float(np.degrees(np.arccos(np.clip(np.trace(Q) / 2 - 0.5, -1, 1))))
    if angle > max_correction_deg:
        return sim, {"levelling": f"rejected ({angle:.1f} deg correction looks wrong)"}
    rot = Q @ sim.rotation
    x = sim.scale * model_c @ rot.T  # levelled, before translation
    xi, gi = x[inliers], gps[inliers]
    # Horizontal: 2-D similarity (yaw + scale + shift); vertical: shift only.
    a, b = xi[:, :2] - xi[:, :2].mean(0), gi[:, :2] - gi[:, :2].mean(0)
    cov = b.T @ a
    u, _d, vt = np.linalg.svd(cov)
    S = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    R2 = u @ S @ vt
    s2 = float(np.trace(np.diag(_d) @ S) / (a**2).sum())
    Rz = np.eye(3)
    Rz[:2, :2] = R2
    scale = sim.scale * s2
    rotation = Rz @ rot
    pred = scale * model_c @ rotation.T
    t = np.empty(3)
    t[:2] = (gi[:, :2] - pred[inliers][:, :2]).mean(0)
    t[2] = float(np.median(gi[:, 2] - pred[inliers][:, 2]))
    return Similarity(scale, rotation, t), {"levelling": method, "tilt_correction_deg": angle}


def apply_to_reconstruction(rec: pycolmap.Reconstruction, sim: Similarity) -> None:
    rec.transform(
        pycolmap.Sim3d(sim.scale, pycolmap.Rotation3d(sim.rotation), sim.translation.reshape(3, 1))
    )


def accuracy_summary(geo: GeorefResult, reproj_px: float, gsd_m: float) -> dict:
    """Plain-language accuracy statement for the report and viewer.

    * relative (within-model) accuracy comes from the image geometry: roughly
      reprojection error x ground sample distance, plus the scale uncertainty
      for long lengths;
    * absolute (on-the-globe) accuracy cannot beat the GPS: with consumer GPS
      the whole model may be shifted by about the residual RMS, which is why
      RTK or ground control points are needed for survey-grade positions.
    """
    rel = max(reproj_px, 0.5) * gsd_m * 2.0
    rms_h = geo.stats["gps_residual_rms_h_m"]
    rms_v = geo.stats["gps_residual_rms_v_m"]
    grade = "consumer-gps"
    if rms_h < 0.1:
        grade = "rtk-grade"
    return {
        "relative_point_m": rel,
        "length_error_percent": 100 * geo.scale_sigma_rel,
        "absolute_h_m": rms_h,
        "absolute_v_m": rms_v,
        "gsd_m": gsd_m,
        "grade": grade,
        "note": (
            "Lengths, areas and volumes measured inside the model are reliable to "
            "about the relative accuracy plus the length error percentage. "
            "Absolute coordinates are only as good as the drone GPS; use RTK or "
            "ground control points for survey-grade positions."
        ),
    }
