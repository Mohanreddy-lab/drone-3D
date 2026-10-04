"""One command from drone video to a measurable, georeferenced model.

    ingest -> keyframes -> SfM -> georeference -> dense -> export -> report

Each stage writes its outputs into the run directory; ``resume=True`` skips
stages whose outputs already exist, so an interrupted run continues instead
of starting again.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pycolmap

from drone3d import dense, export, georef, sfm, telemetry
from drone3d.frames import SelectedFrame, select_keyframes
from drone3d.geo import LocalFrame


@dataclass
class Options:
    overlap: float = 0.85
    max_dim: int = 1600
    focal_35mm: float | None = None
    default_pitch: float = -90.0
    time_offset: float = 0.0
    sync_search_s: float = 1.0
    max_features: int = 6000
    gps_sigma: float = 3.0
    dense: bool = True
    dense_max_dim: int = 1000
    voxel: float = 0.2
    raster_gsd: float = 0.2
    resume: bool = False


class Logger:
    def __init__(self, path: Path, echo: Callable | None = print) -> None:
        self.path = path
        self.echo = echo
        self.t0 = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()

    def __call__(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        if self.echo:
            self.echo(line)

    def close(self) -> None:
        self.fh.close()


def run(
    video: str | Path,
    out_dir: str | Path,
    telemetry_path: str | Path | None = None,
    options: Options | None = None,
    echo: Callable | None = print,
    progress: Callable[[str, float], None] | None = None,
) -> dict:
    opts = options or Options()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "run.log", echo)
    step = progress or (lambda stage, frac: None)
    timings: dict[str, float] = {}
    t_start = time.time()

    def timed(name: str):
        class _T:
            def __enter__(self_inner):
                self_inner.t = time.time()
                step(name, 0.0)
                log(f"== {name} ==")

            def __exit__(self_inner, *exc):
                timings[name] = round(time.time() - self_inner.t, 1)
                step(name, 1.0)

        return _T()

    try:
        # 1. Ingest -----------------------------------------------------------
        with timed("ingest"):
            tel = telemetry.load(telemetry_path) if telemetry_path else None
            frame = tel.frame() if tel else LocalFrame(0.0, 0.0, 0.0)
            if tel:
                log(
                    f"telemetry: {len(tel)} fixes over {tel.t[-1] - tel.t[0]:.1f}s "
                    f"from {tel.source}"
                )
                for note in tel.notes:
                    log(f"telemetry: {note}")
            else:
                log("no telemetry: model will be unscaled and not georeferenced")

        # 2. Keyframes --------------------------------------------------------
        image_dir = out / "images"
        frames_json = out / "work" / "frames.json"
        with timed("keyframes"):
            if opts.resume and frames_json.exists():
                data = json.loads(frames_json.read_text())
                frames = [SelectedFrame(**f) for f in data["frames"]]
                frame_stats = data["stats"]
                log(f"resume: {len(frames)} keyframes")
            else:
                frames, frame_stats = select_keyframes(
                    video,
                    image_dir,
                    tel,
                    frame if tel else None,
                    overlap=opts.overlap,
                    max_dim=opts.max_dim,
                    focal_35mm=opts.focal_35mm,
                    default_pitch=opts.default_pitch,
                    time_offset=opts.time_offset,
                    log=log,
                )
                frames_json.parent.mkdir(parents=True, exist_ok=True)
                frames_json.write_text(
                    json.dumps({"frames": [asdict(f) for f in frames], "stats": frame_stats})
                )
            if len(frames) < 3:
                raise RuntimeError("fewer than 3 usable keyframes; is the video long enough?")

        # 3. SfM ---------------------------------------------------------------
        model_dir = out / "work" / "model"
        with timed("sfm"):
            if opts.resume and (model_dir / "images.bin").exists():
                rec = pycolmap.Reconstruction(model_dir)
                sfm_stats = json.loads((out / "work" / "sfm.json").read_text())
                log("resume: georeferenced model loaded")
                geo_stats = sfm_stats.pop("_georef")
                accuracy = sfm_stats.pop("_accuracy")
            else:
                res = sfm.run_sfm(
                    image_dir,
                    out / "work",
                    frames,
                    focal_35mm=opts.focal_35mm or frame_stats.get("focal_35mm"),
                    max_features=opts.max_features,
                    gps_sigma=opts.gps_sigma,
                    log=log,
                )
                rec, sfm_stats = res.reconstruction, res.stats
                geo_stats, accuracy = None, None

        # 4. Georeference ------------------------------------------------------
        with timed("georeference"):
            if geo_stats is None:
                if tel:
                    g = georef.georeference(
                        rec,
                        frames,
                        tel,
                        frame,
                        threshold=max(3 * opts.gps_sigma, 4.0),
                        search_offset=opts.sync_search_s,
                        initial_offset=opts.time_offset,
                        log=log,
                    )
                    georef.apply_to_reconstruction(rec, g.similarity)
                    geo_stats = g.stats
                    gsd = _gsd(rec)
                    accuracy = georef.accuracy_summary(
                        g, sfm_stats["mean_reprojection_error_px"], gsd
                    )
                else:
                    geo_stats = {"georeferenced": False}
                    accuracy = {
                        "grade": "unscaled",
                        "note": "No GPS: arbitrary scale and orientation.",
                    }
                model_dir.mkdir(parents=True, exist_ok=True)
                rec.write(model_dir)
                (out / "work" / "sfm.json").write_text(
                    json.dumps(
                        sfm_stats | {"_georef": geo_stats, "_accuracy": accuracy}, default=float
                    )
                )

        sparse_xyz, sparse_rgb, _ = sfm.sparse_points(rec)
        export.write_ply(out / "sparse.ply", sparse_xyz, sparse_rgb)

        # 5. Dense -------------------------------------------------------------
        dense_stats: dict = {"method": "skipped"}
        pts, rgb = sparse_xyz, sparse_rgb
        with timed("dense"):
            if opts.dense:
                if opts.resume and (out / "cloud.ply").exists():
                    pts, rgb = export.read_ply(out / "cloud.ply")
                    dense_stats = {"method": "resumed", "points": len(pts)}
                else:
                    pts, rgb, dense_stats = dense.densify(
                        rec,
                        image_dir,
                        out / "work",
                        max_dim=opts.dense_max_dim,
                        voxel=opts.voxel,
                        log=log,
                    )
            export.write_ply(out / "cloud.ply", pts, rgb)

        # 6. Export ------------------------------------------------------------
        with timed("export"):
            names, centers, gps = _camera_table(rec, frames)
            raster_stats: dict = {}
            if tel:
                epsg = export.write_las(out / "cloud.las", frame, pts, rgb)
                export.write_cameras_geojson(out / "cameras.geojson", frame, names, centers, gps)
                raster_stats = export.write_rasters(out, frame, pts, rgb, opts.raster_gsd)
                log(
                    f"exported LAS (EPSG:{epsg}), DSM/orthophoto at {opts.raster_gsd} m, "
                    "camera GeoJSON"
                )
            ground = float(np.percentile(pts[:, 2], 5)) if len(pts) else 0.0

        report = {
            "input": {
                "video": str(video),
                "telemetry": str(telemetry_path) if telemetry_path else None,
                "telemetry_source": tel.source if tel else None,
            },
            "origin": {
                "lat": frame.lat0,
                "lon": frame.lon0,
                "h": frame.h0,
                "frame": "ENU (east, north, up) metres",
            },
            "epsg": frame.utm_epsg() if tel else None,
            "keyframes": frame_stats,
            "sfm": sfm_stats,
            "georeference": geo_stats,
            "dense": dense_stats,
            "rasters": raster_stats,
            "accuracy": accuracy,
            "timings_s": timings | {"total": round(time.time() - t_start, 1)},
            "outputs": sorted(p.name for p in out.iterdir() if p.is_file()),
        }
        (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
        scene = {
            "origin": report["origin"],
            "epsg": report["epsg"],
            "georeferenced": bool(tel),
            "cloud": "cloud.ply",
            "sparse": "sparse.ply",
            "points": int(len(pts)),
            "ground_z": ground,
            "cameras": [
                {"name": n, "center": [round(float(v), 3) for v in c]}
                for n, c in zip(names, centers, strict=True)
            ],
            "accuracy": accuracy,
            "report": "report.json",
            "downloads": [
                p
                for p in (
                    "cloud.las",
                    "cloud.ply",
                    "dsm.tif",
                    "ortho.png",
                    "dsm_hillshade.png",
                    "cameras.geojson",
                    "report.json",
                )
                if (out / p).exists()
            ],
        }
        (out / "scene.json").write_text(json.dumps(scene, default=float))
        log(f"done in {report['timings_s']['total']:.0f}s -> {out}")
        return report
    finally:
        log.close()


def _camera_table(rec: pycolmap.Reconstruction, frames: list[SelectedFrame]):
    by_name = {f.name: f for f in frames}
    imgs = sorted((img for img in rec.images.values() if img.has_pose), key=lambda i: i.name)
    names = [i.name for i in imgs]
    centers = np.array([i.projection_center() for i in imgs])
    gps = None
    if all(by_name.get(n) and by_name[n].enu is not None for n in names):
        gps = np.array([by_name[n].enu for n in names])
    return names, centers, gps


def _gsd(rec: pycolmap.Reconstruction) -> float:
    """Median ground sample distance (m/pixel) over registered images."""
    cam = rec.cameras[next(iter(rec.cameras))]
    f = float(cam.mean_focal_length())
    depths = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        m = np.asarray(img.cam_from_world().matrix())
        pids = [p.point3D_id for p in img.points2D if p.has_point3D()][:200]
        if not pids:
            continue
        xyz = np.array([rec.points3D[p].xyz for p in pids])
        z = (xyz @ m[:, :3].T + m[:, 3])[:, 2]
        depths.append(np.median(z[z > 0]) if (z > 0).any() else math.nan)
    d = float(np.nanmedian(depths)) if depths else math.nan
    return d / f
