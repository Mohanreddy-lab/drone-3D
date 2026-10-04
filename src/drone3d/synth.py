"""Synthetic drone survey generator -- a test site with known ground truth.

Renders a textured site (ground, two buildings, a stockpile) from a lawnmower
flight, encodes it to H.264 like a real drone would, and writes a DJI-style
``.SRT`` with *noisy* consumer-grade GPS. Because every dimension of the site
is known, the pipeline's accuracy can be measured instead of eyeballed:

* building A: 20 x 12 m footprint, 8 m tall
* building B: 15 x 15 m footprint, 12 m tall
* stockpile: square pyramid, 16 x 16 m base, 6 m tall -> 512 m^3

The renderer is a vectorised NumPy ray caster -- no GPU, no 3D engine.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from drone3d.geo import LocalFrame, look_rotation

ORIGIN_LAT, ORIGIN_LON = 17.385044, 78.486671  # site origin (Hyderabad)


@dataclass
class Box:
    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    height: float


@dataclass
class Pyramid:
    name: str
    cx: float
    cy: float
    base: float
    height: float

    @property
    def volume(self) -> float:
        return self.base**2 * self.height / 3.0


@dataclass
class Site:
    boxes: list[Box] = field(
        default_factory=lambda: [
            Box("building_A", -32.0, -12.0, -14.0, -2.0, 8.0),
            Box("building_B", 8.0, 23.0, 6.0, 21.0, 12.0),
        ]
    )
    pyramids: list[Pyramid] = field(
        default_factory=lambda: [Pyramid("stockpile", -5.0, 28.0, 16.0, 6.0)]
    )
    half_x: float = 110.0
    half_y: float = 110.0

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = np.zeros(np.broadcast(x, y).shape)
        for b in self.boxes:
            inside = (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
            z = np.where(inside, np.maximum(z, b.height), z)
        for p in self.pyramids:
            r = np.maximum(np.abs(x - p.cx), np.abs(y - p.cy)) / (p.base / 2)
            z = np.maximum(z, np.where(r < 1, p.height * (1 - r), 0.0))
        return z


# ---------------------------------------------------------------------------
# Geometry: every surface is a parallelogram or triangle (origin + 2 edges).


@dataclass
class Prim:
    p0: np.ndarray
    a: np.ndarray
    b: np.ndarray
    tri: bool
    color: np.ndarray
    seed: int
    contrast: float = 0.55


def _site_prims(site: Site) -> list[Prim]:
    prims: list[Prim] = []
    hx, hy = site.half_x, site.half_y
    prims.append(
        Prim(
            np.array([-hx, -hy, 0.0]),
            np.array([2 * hx, 0, 0.0]),
            np.array([0, 2 * hy, 0.0]),
            False,
            np.array([0.42, 0.50, 0.30]),
            1,
            0.65,
        )
    )
    seed = 10
    palette = [np.array([0.62, 0.58, 0.55]), np.array([0.70, 0.45, 0.35])]
    for i, bx in enumerate(site.boxes):
        x0, x1, y0, y1, h = bx.x0, bx.x1, bx.y0, bx.y1, bx.height
        roof = palette[i % 2]
        wall = np.array([0.80, 0.76, 0.68])
        quads = [
            ([x0, y0, h], [x1 - x0, 0, 0], [0, y1 - y0, 0], roof),  # roof
            ([x0, y0, 0], [x1 - x0, 0, 0], [0, 0, h], wall),  # south
            ([x0, y1, 0], [x1 - x0, 0, 0], [0, 0, h], wall),  # north
            ([x0, y0, 0], [0, y1 - y0, 0], [0, 0, h], wall),  # west
            ([x1, y0, 0], [0, y1 - y0, 0], [0, 0, h], wall),  # east
        ]
        for p0, a, b, c in quads:
            prims.append(
                Prim(np.array(p0, float), np.array(a, float), np.array(b, float), False, c, seed)
            )
            seed += 1
    for p in site.pyramids:
        s = p.base / 2
        apex = np.array([p.cx, p.cy, p.height])
        corners = [
            np.array([p.cx - s, p.cy - s, 0.0]),
            np.array([p.cx + s, p.cy - s, 0.0]),
            np.array([p.cx + s, p.cy + s, 0.0]),
            np.array([p.cx - s, p.cy + s, 0.0]),
        ]
        for k in range(4):
            c0, c1 = corners[k], corners[(k + 1) % 4]
            prims.append(
                Prim(c0, c1 - c0, apex - c0, True, np.array([0.66, 0.56, 0.40]), seed, 0.7)
            )
            seed += 1
    return prims


TEXEL = 0.04  # metres per texture pixel


def _texture(width_m: float, height_m: float, seed: int, roughness: float = 0.8) -> np.ndarray:
    """Multi-scale random texture (float32, ~0..1) covering a surface.

    Built by upsampling random grids from ~8 m down to texel scale, so it has
    corners at every altitude and never repeats -- a repeated texture would
    give the feature matcher false correspondences.
    """
    rng = np.random.default_rng(seed)
    w = max(int(np.ceil(width_m / TEXEL)), 8)
    h = max(int(np.ceil(height_m / TEXEL)), 8)
    out = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    wavelength = 8.0
    while wavelength >= TEXEL * 1.5:
        gw = max(int(np.ceil(width_m / wavelength)) + 2, 2)
        gh = max(int(np.ceil(height_m / wavelength)) + 2, 2)
        grid = rng.random((gh, gw), dtype=np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        norm += amp
        amp *= roughness
        wavelength /= 2.0
    out /= norm
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


_TEX_CACHE: dict[tuple[int, str], np.ndarray] = {}


def _prim_texture(p: Prim) -> np.ndarray:
    key = (p.seed, "tex")
    if key not in _TEX_CACHE:
        _TEX_CACHE[key] = _texture(float(np.linalg.norm(p.a)), float(np.linalg.norm(p.b)), p.seed)
    return _TEX_CACHE[key]


def _mix_texture(p: Prim) -> np.ndarray:
    key = (p.seed, "mix")
    if key not in _TEX_CACHE:
        la, lb = float(np.linalg.norm(p.a)), float(np.linalg.norm(p.b))
        rng = np.random.default_rng(p.seed + 5000)
        grid = rng.random((int(la / 20) + 2, int(lb / 20) + 2), dtype=np.float32)
        shape = _prim_texture(p).shape
        _TEX_CACHE[key] = cv2.resize(grid.T, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return _TEX_CACHE[key]


def _sample(tex: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """Bilinear lookup of N points (cv2.remap caps each dimension at 32767)."""
    n = len(mx)
    cols = 4096
    rows = -(-n // cols)
    pad = rows * cols - n
    mx2 = np.pad(mx, (0, pad)).reshape(rows, cols)
    my2 = np.pad(my, (0, pad)).reshape(rows, cols)
    return cv2.remap(tex, mx2, my2, cv2.INTER_LINEAR).reshape(-1)[:n]


SUN = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])


@dataclass
class CameraSpec:
    width: int = 960
    height: int = 540
    focal_35mm: float = 24.0  # DJI-style 35 mm equivalent focal length

    @property
    def fx(self) -> float:
        return self.focal_35mm / 36.0 * self.width

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.width / 2], [0, self.fx, self.height / 2], [0, 0, 1]])


def render(prims: list[Prim], cam: CameraSpec, center: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    """Ray-cast one RGB frame (uint8). ``R_wc`` maps camera axes to world."""
    w, h = cam.width, cam.height
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_c = np.stack([(xs - w / 2) / cam.fx, (ys - h / 2) / cam.fx, np.ones_like(xs)], -1)
    d = dirs_c.reshape(-1, 3) @ R_wc.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = len(d)
    best_t = np.full(n, np.inf)
    best_id = np.full(n, -1)
    best_u = np.zeros(n)
    best_v = np.zeros(n)
    for k, p in enumerate(prims):
        pvec = np.cross(d, p.b)
        det = pvec @ p.a
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = center - p.p0
        u = (pvec @ tvec) * inv
        qvec = np.cross(tvec, p.a)
        v = (d @ qvec) * inv
        t = (qvec @ p.b) * inv
        inside = ok & (u >= 0) & (v >= 0) & (t > 0.01) & (t < best_t)
        inside &= (u + v <= 1) if p.tri else ((u <= 1) & (v <= 1))
        best_t = np.where(inside, t, best_t)
        best_id = np.where(inside, k, best_id)
        best_u = np.where(inside, u, best_u)
        best_v = np.where(inside, v, best_v)
    img = np.empty((n, 3), np.float32)
    img[:] = (0.62, 0.74, 0.90)  # sky
    for k, p in enumerate(prims):
        m = best_id == k
        if not m.any():
            continue
        tex = _prim_texture(p)
        mx = (best_u[m] * (tex.shape[1] - 1)).astype(np.float32)
        my = (best_v[m] * (tex.shape[0] - 1)).astype(np.float32)
        val_tex = _sample(tex, mx, my)
        normal = np.cross(p.a, p.b)
        normal /= np.linalg.norm(normal)
        if normal[2] < 0 or (abs(normal[2]) < 1e-6 and normal @ (center - p.p0) < 0):
            normal = -normal
        shade = 0.45 + 0.55 * max(float(normal @ SUN), 0.0)
        val = (1 - p.contrast) + p.contrast * 1.25 * val_tex
        if k == 0:  # ground: mix grass and soil by low-frequency noise
            mix = _sample(_mix_texture(p), mx, my)
            mix = mix[:, None]
            col = p.color * (1 - mix) + np.array([0.55, 0.45, 0.33]) * mix
        else:
            col = np.broadcast_to(p.color, (int(m.sum()), 3))
        img[m] = col * (val * shade)[:, None]
    img = img.reshape(h, w, 3)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Flight plan


def lawnmower(
    lines_x: tuple[float, ...] = (-30.0, 0.0, 30.0),
    y_extent: tuple[float, float] = (-45.0, 50.0),
    altitude: float = 40.0,
    step: float = 0.05,
) -> np.ndarray:
    """Dense polyline (x, y, z): straight survey lines joined by U-turns."""
    pts: list[np.ndarray] = []
    for i, x in enumerate(lines_x):
        y0, y1 = y_extent if i % 2 == 0 else y_extent[::-1]
        ys = np.arange(y0, y1, step if y1 > y0 else -step)
        pts.append(np.stack([np.full_like(ys, x), ys, np.full_like(ys, altitude)], 1))
        if i + 1 < len(lines_x):
            r = (lines_x[i + 1] - x) / 2
            cx = x + r
            ang = np.arange(0, np.pi, step / abs(r))
            if y1 > y0:
                arc = np.stack([cx - r * np.cos(ang), y1 + abs(r) * np.sin(ang)], 1)
            else:
                arc = np.stack([cx - r * np.cos(ang), y1 - abs(r) * np.sin(ang)], 1)
            pts.append(np.column_stack([arc, np.full(len(arc), altitude)]))
    return np.concatenate(pts)


def sample_path(
    path: np.ndarray, speed: float, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    duration = s[-1] / speed
    t = np.arange(0, duration, 1.0 / fps)
    sq = t * speed
    pos = np.stack([np.interp(sq, s, path[:, k]) for k in range(3)], 1)
    ahead = np.stack([np.interp(np.minimum(sq + 2.0, s[-1]), s, path[:, k]) for k in range(2)], 1)
    behind = np.stack([np.interp(np.maximum(sq - 2.0, 0), s, path[:, k]) for k in range(2)], 1)
    dv = ahead - behind
    yaw = np.degrees(np.arctan2(dv[:, 0], dv[:, 1]))
    return t, pos, yaw


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    path: Path, t: np.ndarray, lat, lon, rel_alt, yaw, pitch, focal: float, fps: float
) -> None:
    lines = []
    for i in range(len(t)):
        t1 = t[i] + 1.0 / fps
        lines.append(
            f"{i + 1}\n{_srt_time(t[i])} --> {_srt_time(t1)}\n"
            f'<font size="28">FrameCnt: {i + 1}, DiffTime: {int(1000 / fps)}ms\n'
            f"2026-10-04 10:00:{t[i]:06.3f}\n"
            f"[iso : 100] [shutter : 1/1000.0] [fnum : 2.8] [ev : 0] "
            f"[focal_len : {focal:.2f}] [latitude: {lat[i]:.7f}] "
            f"[longitude: {lon[i]:.7f}] [rel_alt: {rel_alt[i]:.3f} "
            f"abs_alt: {rel_alt[i] + 520.0:.3f}] [gb_yaw: {yaw[i]:.1f} "
            f"gb_pitch: {pitch[i]:.1f} gb_roll: 0.0] </font>\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def gps_noise(n: int, fps: float, sigma_h: float, sigma_v: float, seed: int) -> np.ndarray:
    """Consumer GPS error: a slowly wandering bias plus a little jitter."""
    rng = np.random.default_rng(seed)
    tau = 20.0  # seconds; correlation time of the wander
    a = np.exp(-1.0 / (fps * tau))
    bias = np.zeros((n, 3))
    sig = np.array([sigma_h, sigma_h, sigma_v])
    bias[0] = rng.normal(0, sig)
    for i in range(1, n):
        bias[i] = a * bias[i - 1] + np.sqrt(1 - a * a) * rng.normal(0, sig)
    return bias + rng.normal(0, sig * 0.15, (n, 3))


def generate(
    out_dir: str | Path,
    *,
    width: int = 960,
    height: int = 540,
    fps: float = 10.0,
    speed: float = 5.0,
    altitude: float = 40.0,
    gimbal_pitch: float = -75.0,
    lines_x: tuple[float, ...] = (-30.0, 0.0, 30.0),
    y_extent: tuple[float, float] = (-45.0, 50.0),
    gps_sigma_h: float = 1.5,
    gps_sigma_v: float = 2.5,
    blur_fraction: float = 0.06,
    max_seconds: float | None = None,
    seed: int = 7,
    log=print,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    site = Site()
    prims = _site_prims(site)
    cam = CameraSpec(width, height)
    t, pos, yaw = sample_path(lawnmower(lines_x, y_extent, altitude), speed, fps)
    if max_seconds is not None:
        keep = t <= max_seconds
        t, pos, yaw = t[keep], pos[keep], yaw[keep]
    rng = np.random.default_rng(seed)
    video_path = out / "flight.mp4"
    writer = _VideoSink(video_path, width, height, fps)
    blurred = []
    log(f"rendering {len(t)} frames at {width}x{height}...")
    for i in range(len(t)):
        R = look_rotation(float(yaw[i]), gimbal_pitch)
        img = render(prims, cam, pos[i], R)
        if rng.random() < blur_fraction:  # simulate wind gusts / jerky gimbal
            k = int(rng.integers(9, 17))
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)
            blurred.append(i)
        noise = rng.normal(0, 2.0, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(img)
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(t)}")
    writer.close()

    noisy = pos + gps_noise(len(t), fps, gps_sigma_h, gps_sigma_v, seed)
    frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON, 0.0)
    lat, lon, _ = frame.to_geodetic(noisy)
    pitch = np.full(len(t), gimbal_pitch)
    srt_path = out / "flight.SRT"
    write_srt(srt_path, t, lat, lon, noisy[:, 2], yaw, pitch, cam.focal_35mm, fps)

    truth = {
        "origin": {"lat": ORIGIN_LAT, "lon": ORIGIN_LON, "h": 0.0},
        "camera": {"width": width, "height": height, "fx": cam.fx, "focal_35mm": cam.focal_35mm},
        "boxes": [asdict(b) for b in site.boxes],
        "pyramids": [asdict(p) | {"volume": p.volume} for p in site.pyramids],
        "gps_sigma_h": gps_sigma_h,
        "gps_sigma_v": gps_sigma_v,
        "blurred_frames": blurred,
        "fps": fps,
        "true_positions": pos.round(4).tolist(),
        "times": t.round(4).tolist(),
    }
    (out / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    log(f"wrote {video_path}, {srt_path}, truth.json")
    return {"video": str(video_path), "telemetry": str(srt_path), "truth": str(out / "truth.json")}


class _VideoSink:
    """H.264 via ffmpeg when available (what real drones record), else mp4v."""

    def __init__(self, path: Path, w: int, h: int, fps: float) -> None:
        self.proc = None
        self.cv = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ],
                stdin=subprocess.PIPE,
            )
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, rgb: np.ndarray) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.write(rgb.tobytes())
        else:
            self.cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the synthetic video")
        else:
            self.cv.release()
