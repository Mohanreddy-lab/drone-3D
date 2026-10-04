"""Command-line entry point: ``drone3d run | demo | synth | evaluate | serve | view``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--overlap", type=float, default=0.85, help="target forward overlap (0-1)")
    p.add_argument("--max-dim", type=int, default=1600, help="keyframe long side in pixels")
    p.add_argument("--focal-35mm", type=float, default=None, help="35 mm equivalent focal length")
    p.add_argument("--pitch", type=float, default=-90.0, help="gimbal pitch if not in telemetry")
    p.add_argument("--time-offset", type=float, default=0.0, help="telemetry clock offset (s)")
    p.add_argument(
        "--sync-search", type=float, default=1.0, help="auto clock-sync search range (s)"
    )
    p.add_argument("--gps-sigma", type=float, default=3.0, help="GPS 1-sigma error (m)")
    p.add_argument("--no-dense", action="store_true", help="sparse model only (fast)")
    p.add_argument("--voxel", type=float, default=0.2, help="dense fusion voxel (m)")
    p.add_argument("--gsd", type=float, default=0.2, help="DSM/orthophoto cell size (m)")
    p.add_argument("--resume", action="store_true", help="reuse finished stages")


def _options(a: argparse.Namespace):
    from drone3d.pipeline import Options

    return Options(
        overlap=a.overlap,
        max_dim=a.max_dim,
        focal_35mm=a.focal_35mm,
        default_pitch=a.pitch,
        time_offset=a.time_offset,
        sync_search_s=a.sync_search,
        gps_sigma=a.gps_sigma,
        dense=not a.no_dense,
        voxel=a.voxel,
        raster_gsd=a.gsd,
        resume=a.resume,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drone3d", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="video + telemetry -> georeferenced 3D model")
    r.add_argument("video")
    r.add_argument("-t", "--telemetry", help="DJI .SRT or CSV flight log")
    r.add_argument("-o", "--out", required=True)
    _add_run_options(r)

    d = sub.add_parser("demo", help="generate a synthetic survey, reconstruct it, score it")
    d.add_argument("-o", "--out", default="demo_run")
    d.add_argument("--quick", action="store_true", help="short low-res flight (~2 min total)")
    _add_run_options(d)

    s = sub.add_parser("synth", help="generate a synthetic drone video + SRT with ground truth")
    s.add_argument("out")
    s.add_argument("--quick", action="store_true")

    e = sub.add_parser("evaluate", help="score a run against synthetic ground truth")
    e.add_argument("run")
    e.add_argument("--truth", required=True)

    sv = sub.add_parser("serve", help="web app: upload, process, view and measure")
    sv.add_argument("--data", default="drone3d_data")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    v = sub.add_parser("view", help="open the 3D viewer on a finished run")
    v.add_argument("run")
    v.add_argument("--port", type=int, default=8001)

    a = ap.parse_args(argv)

    if a.cmd == "run":
        from drone3d.pipeline import run

        run(a.video, a.out, a.telemetry, _options(a))
        print(f"\nview it:  drone3d view {a.out}")
        return 0

    if a.cmd == "synth":
        from drone3d import synth

        synth.generate(a.out, **(QUICK if a.quick else {}))
        return 0

    if a.cmd == "demo":
        from drone3d import synth
        from drone3d.evaluate import evaluate
        from drone3d.pipeline import run

        out = Path(a.out)
        data = synth.generate(out / "input", **(QUICK if a.quick else {}))
        run(data["video"], out / "result", data["telemetry"], _options(a))
        if not a.no_dense:
            res = evaluate(out / "result", data["truth"])
            print_evaluation(res)
        print(f"\nview it:  drone3d view {out / 'result'}")
        return 0

    if a.cmd == "evaluate":
        from drone3d.evaluate import evaluate

        print_evaluation(evaluate(a.run, a.truth))
        return 0

    if a.cmd == "serve":
        from drone3d.server import serve

        serve(Path(a.data), a.host, a.port)
        return 0

    if a.cmd == "view":
        from drone3d.server import view

        view(Path(a.run), a.port)
        return 0
    return 1


QUICK = {
    "width": 640,
    "height": 360,
    "fps": 6.0,
    "speed": 6.0,
    "lines_x": (-20.0, 10.0),
    "y_extent": (-35.0, 45.0),
}


def print_evaluation(res: dict) -> None:
    s, c = res["surface"], res["camera_positions"]
    print("\n=== accuracy vs ground truth ===")
    off = s["model_offset_m"]
    print(
        f"absolute: model sits {np.hypot(off[0], off[1]):.2f} m horizontally and {off[2]:+.2f} m "
        "vertically from the truth (consumer-GPS limit)"
    )
    print(
        f"relative: surface RMSE {s['vertical_rmse_m']:.3f} m, "
        f"{s['within_25cm_percent']:.0f}% of points within 25 cm of the true shape"
    )
    print(
        f"cameras: absolute RMSE {c['abs_h_rmse_m']:.2f} m horizontal, "
        f"{c['abs_v_rmse_m']:.2f} m vertical "
        f"over {c['cameras']} keyframes"
    )
    for o in res["objects"]:
        parts = []
        for k, tv in o["true"].items():
            mv = o["measured"].get(k)
            if mv is None:
                continue
            err = 100 * (mv - tv) / tv if tv else float("nan")
            parts.append(f"{k} {mv:.2f} (true {tv:.2f}, {err:+.1f}%)")
        print(f"{o['name']}: " + "; ".join(parts))
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
