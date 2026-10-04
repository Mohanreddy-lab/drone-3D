"""Keyframe selection: from thousands of video frames to the few that matter.

A 30 fps drone video moving at 5 m/s produces a frame every 17 cm -- far more
overlap than photogrammetry needs, and every redundant frame costs matching
time quadratically. We keep a frame each time the camera has moved far enough
for the target forward overlap, and inside each window pick the *sharpest*
candidate, so motion-blurred frames are skipped instead of reconstructed.

"Moved far enough" is measured in metres from GPS when telemetry exists, and
in image pixels from optical flow when it does not -- the same selection logic
runs on either progress signal.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drone3d.geo import LocalFrame
from drone3d.telemetry import Telemetry


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class SelectedFrame:
    name: str
    index: int
    time: float
    sharpness: float
    enu: tuple[float, float, float] | None = None
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = VideoInfo(
        str(path),
        float(fps),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def _iter_frames(path: str | Path):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian -- drops sharply under motion blur."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def analyse(
    path: str | Path, with_flow: bool, analysis_width: int = 480, log: Callable = print
) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1 over the video: per-frame sharpness and (optionally) image motion."""
    sharp: list[float] = []
    flow: list[float] = []
    prev = None
    prev_pts = None
    for i, frame in _iter_frames(path):
        scale = analysis_width / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp.append(sharpness(gray))
        if with_flow:
            motion = 0.0
            if prev is not None and prev_pts is not None and len(prev_pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, prev_pts, None)
                good = st.reshape(-1) == 1
                if good.sum() >= 8:
                    d = np.linalg.norm((nxt - prev_pts).reshape(-1, 2)[good], axis=1)
                    motion = float(np.median(d))
            flow.append(motion)
            prev = gray
            prev_pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 8)
        if (i + 1) % 500 == 0:
            log(f"  analysed {i + 1} frames")
    return np.asarray(sharp), np.asarray(flow)


def footprint(alt: float, vfov: float, hfov: float, pitch_deg: float) -> float:
    """Shorter side (m) of the ground footprint for a camera at ``alt``.

    For an oblique gimbal the near edge is closer than the far edge; using the
    near-edge geometry keeps the overlap estimate conservative.
    """
    alt = max(alt, 5.0)
    off_nadir = math.radians(90.0 + pitch_deg)  # 0 = straight down
    near = off_nadir - vfov / 2
    far = min(off_nadir + vfov / 2, math.radians(80))
    along = alt * (math.tan(far) - math.tan(near))
    across = 2 * alt * math.tan(hfov / 2) / max(math.cos(off_nadir), 0.2)
    return float(min(along, across))


def choose(
    progress: np.ndarray,
    spacing: np.ndarray | float,
    sharp: np.ndarray,
    yaw: np.ndarray | None = None,
    yaw_step: float = 20.0,
    window_frac: float = 0.3,
    blur_ratio: float = 0.55,
    smooth_frames: int = 31,
) -> list[int]:
    """Indices of keyframes along a monotonic ``progress`` signal.

    * a new keyframe is due once progress since the last one reaches
      ``spacing`` (or heading changes by ``yaw_step`` degrees);
    * among frames in the following window (``window_frac`` of the spacing)
      the sharpest wins;
    * frames below ``blur_ratio`` x the local median sharpness are never
      chosen if anything else in the window is acceptable.
    """
    n = len(progress)
    if n == 0:
        return []
    spacing_arr = np.broadcast_to(np.asarray(spacing, float), (n,))
    k = max(3, smooth_frames | 1)
    pad = np.pad(sharp, k // 2, mode="edge")
    local_med = np.array([np.median(pad[i : i + k]) for i in range(n)])
    ok = sharp >= blur_ratio * local_med

    def best_in(a: int, b: int) -> int:
        idx = np.arange(a, min(b, n))
        cand = idx[ok[idx]] if ok[idx].any() else idx
        return int(cand[np.argmax(sharp[cand])])

    # First keyframe: sharpest frame near the start.
    first_end = int(np.searchsorted(progress, progress[0] + window_frac * spacing_arr[0]))
    chosen = [best_in(0, max(first_end, 1) + 1)]
    while True:
        last = chosen[-1]
        target = progress[last] + spacing_arr[last]
        j = int(np.searchsorted(progress, target))
        if yaw is not None:
            dyaw = np.abs((yaw[last + 1 :] - yaw[last] + 180) % 360 - 180)
            turn = np.nonzero(dyaw >= yaw_step)[0]
            if len(turn) and last + 1 + turn[0] < j:
                j = last + 1 + int(turn[0])
        if j >= n:
            break
        win_end = int(np.searchsorted(progress, progress[j] + window_frac * spacing_arr[j]))
        pick = best_in(j, max(win_end, j + 1) + 1)
        if pick <= last:
            pick = j
        chosen.append(pick)
    return chosen


def select_keyframes(
    video: str | Path,
    out_dir: str | Path,
    telemetry: Telemetry | None,
    frame: LocalFrame | None,
    *,
    overlap: float = 0.85,
    max_dim: int = 1600,
    focal_35mm: float | None = None,
    default_pitch: float = -90.0,
    time_offset: float = 0.0,
    log: Callable = print,
) -> tuple[list[SelectedFrame], dict]:
    """Analyse the video, choose keyframes, and write them as JPEGs."""
    info = probe(video)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log(f"video: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.frame_count} frames")

    use_gps = telemetry is not None and len(telemetry) >= 2
    sharp, flow = analyse(video, with_flow=not use_gps, log=log)
    n = len(sharp)
    times = np.arange(n) / info.fps
    f35 = focal_35mm or (telemetry.focal_35mm if telemetry else None) or 24.0
    fx = f35 / 36.0 * info.width
    hfov = 2 * math.atan(info.width / 2 / fx)
    vfov = 2 * math.atan(info.height / 2 / fx)

    enu = None
    yaw = None
    stats: dict = {"frames_total": n, "fps": info.fps, "focal_35mm": f35}
    if use_gps:
        assert telemetry is not None
        enu = telemetry.positions_at(times, frame, time_offset)
        step = np.linalg.norm(np.diff(enu, axis=0), axis=1)
        progress = np.concatenate([[0.0], np.cumsum(step)])
        pitch = telemetry.interp_optional(telemetry.gimbal_pitch, times + time_offset)
        pitch = pitch if pitch is not None else np.full(n, default_pitch)
        alts = np.maximum(enu[:, 2], 5.0)
        spacing = np.array(
            [(1 - overlap) * footprint(a, vfov, hfov, p) for a, p in zip(alts, pitch, strict=True)]
        )
        yaw = telemetry.interp_optional(telemetry.yaw, times + time_offset)
        stats["selection"] = "gps-distance"
        stats["median_spacing_m"] = float(np.median(spacing))
    else:
        progress = np.cumsum(flow) if len(flow) else np.zeros(n)
        analysis_h = 480 * info.height / info.width
        spacing = (1 - overlap) * analysis_h
        stats["selection"] = "optical-flow"
        stats["median_spacing_px"] = float(spacing)

    idx = choose(progress, spacing, sharp, yaw)
    keep = set(idx)
    log(f"selected {len(idx)} keyframes from {n} frames ({n / max(len(idx), 1):.1f}x reduction)")

    selected: list[SelectedFrame] = []
    scale = min(1.0, max_dim / max(info.width, info.height))
    for i, img in _iter_frames(video):
        if i not in keep:
            continue
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sf = SelectedFrame(name, i, float(times[i]), float(sharp[i]))
        if enu is not None and frame is not None:
            e = enu[i]
            sf.enu = (float(e[0]), float(e[1]), float(e[2]))
            lat, lon, h = frame.to_geodetic(e)
            sf.lat, sf.lon, sf.alt = float(lat), float(lon), float(h)
        selected.append(sf)

    stats.update(
        frames_selected=len(selected),
        reduction=n / max(len(selected), 1),
        blur_rejected=int(np.sum(sharp < 0.55 * np.median(sharp))),
        image_scale=scale,
        hfov_deg=math.degrees(hfov),
    )
    write_frames_csv(out.parent / "frames.csv", selected)
    return selected, stats


def write_frames_csv(path: Path, frames: list[SelectedFrame]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "index", "time", "sharpness", "lat", "lon", "alt", "e", "n", "u"])
        for f in frames:
            e = f.enu or (None, None, None)
            w.writerow(
                [f.name, f.index, f"{f.time:.3f}", f"{f.sharpness:.1f}", f.lat, f.lon, f.alt, *e]
            )
