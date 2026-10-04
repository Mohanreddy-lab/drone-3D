"""Structure-from-Motion on the selected keyframes (COLMAP via pycolmap).

GPS is used twice before any geometry is solved:

1. as a *focal-length prior* (35 mm equivalent from the SRT) so the camera
   model starts close to the truth instead of guessing; and
2. as *spatial pairing* -- each frame is matched only against its neighbours
   in the sequence and frames within a GPS radius. That is O(n) pairs instead
   of the O(n^2) exhaustive matching, which is what makes long flights
   tractable on a laptop CPU.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from drone3d.frames import SelectedFrame

# COLMAP logs every registration step at INFO; keep the console readable.
pycolmap.logging.minloglevel = 2


@dataclass
class SfMResult:
    reconstruction: pycolmap.Reconstruction
    stats: dict


def run_sfm(
    image_dir: Path,
    work_dir: Path,
    frames: list[SelectedFrame],
    *,
    focal_35mm: float | None,
    max_features: int = 6000,
    seq_overlap: int = 8,
    spatial_radius: float | None = None,
    gps_sigma: float = 3.0,
    use_gps_prior_ba: bool = True,
    threads: int = -1,
    log: Callable = print,
) -> SfMResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "database.db"
    if db_path.exists():
        db_path.unlink()
    names = [f.name for f in frames]

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    if focal_35mm:
        import cv2

        h, w = cv2.imread(str(image_dir / names[0])).shape[:2]
        f = focal_35mm / 36.0 * w
        reader.camera_params = f"{f},{w / 2},{h / 2},0"
    extract = pycolmap.FeatureExtractionOptions()
    extract.sift.max_num_features = max_features
    extract.num_threads = threads
    log(f"extracting SIFT features from {len(names)} images...")
    pycolmap.extract_features(
        db_path,
        image_dir,
        image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extract,
        device=pycolmap.Device.auto,
    )

    has_gps = all(f.enu is not None for f in frames)
    if has_gps:
        _write_pose_priors(db_path, frames, gps_sigma)

    log("matching (sequential neighbours)...")
    seq = pycolmap.SequentialPairingOptions()
    seq.overlap = seq_overlap
    seq.quadratic_overlap = True
    pycolmap.match_sequential(db_path, pairing_options=seq, device=pycolmap.Device.auto)
    if has_gps:
        log("matching (GPS spatial neighbours -- catches adjacent survey lines)...")
        sp = pycolmap.SpatialPairingOptions()
        sp.max_num_neighbors = 30
        sp.ignore_z = False
        if spatial_radius:
            sp.max_distance = spatial_radius
        pycolmap.match_spatial(db_path, pairing_options=sp, device=pycolmap.Device.auto)
    elif len(names) <= 120:
        log("no GPS: exhaustive matching")
        pycolmap.match_exhaustive(db_path, device=pycolmap.Device.auto)

    sparse_dir = work_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = threads
    opts.ba_refine_principal_point = False
    opts.multiple_models = True
    if has_gps and use_gps_prior_ba:
        opts.use_prior_position = True
        opts.use_robust_loss_on_prior_position = True
    log("incremental mapping + bundle adjustment...")
    models = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir, opts)
    if not models:
        raise RuntimeError(
            "reconstruction failed: no image pair could be initialised. "
            "Check overlap (fly slower / lower) and image sharpness."
        )
    best_id = max(models, key=lambda k: models[k].num_reg_images())
    rec = models[best_id]
    stats = {
        "images_input": len(names),
        "images_registered": rec.num_reg_images(),
        "models": len(models),
        "points3D": rec.num_points3D(),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "camera": rec.cameras[next(iter(rec.cameras))].todict()["params"].tolist(),
        "gps_prior_ba": bool(has_gps and use_gps_prior_ba),
    }
    log(
        f"registered {stats['images_registered']}/{len(names)} images, "
        f"{stats['points3D']} points, reprojection {stats['mean_reprojection_error_px']:.2f}px"
    )
    return SfMResult(rec, stats)


def _write_pose_priors(db_path: Path, frames: list[SelectedFrame], sigma: float) -> None:
    by_name = {f.name: f for f in frames}
    db = pycolmap.Database.open(db_path)
    try:
        cov = np.diag([sigma**2, sigma**2, (1.5 * sigma) ** 2])
        for image in db.read_all_images():
            f = by_name.get(image.name)
            if f is None or f.enu is None:
                continue
            prior = pycolmap.PosePrior()
            prior.corr_data_id = image.data_id
            prior.position = np.asarray(f.enu, float)
            prior.position_covariance = cov
            prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            db.write_pose_prior(prior)
    finally:
        db.close()


def camera_centers(rec: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    return {
        img.name: np.asarray(img.projection_center()) for img in rec.images.values() if img.has_pose
    }


def sparse_points(rec: pycolmap.Reconstruction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = rec.points3D
    xyz = np.array([p.xyz for p in pts.values()]) if pts else np.zeros((0, 3))
    rgb = np.array([p.color for p in pts.values()], np.uint8) if pts else np.zeros((0, 3), np.uint8)
    err = np.array([p.error for p in pts.values()]) if pts else np.zeros(0)
    return xyz, rgb, err
