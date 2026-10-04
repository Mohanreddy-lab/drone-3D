"""Flight telemetry: parse, clean, and interpolate to video frame times.

Supported inputs
----------------
* DJI ``.SRT`` subtitle side-cars, both the modern bracketed style
  (``[latitude: 17.38] [longitude: 78.48] [rel_alt: 40.0 abs_alt: 560.1]``)
  and the older ``GPS(lon,lat,alt) BAROMETER:h`` style.
* Generic CSV with a header containing ``time``/``t`` (seconds from video
  start), ``lat``/``latitude``, ``lon``/``longitude`` and ``alt``/``altitude``/
  ``rel_alt``, plus optional ``yaw`` and ``gimbal_pitch``. Exports from
  flight-log tools (Airdata, Mission Planner) map onto this with a rename.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone3d.geo import LocalFrame


@dataclass
class Telemetry:
    t: np.ndarray  # seconds from start of video
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray  # metres above take-off (relative altitude)
    yaw: np.ndarray | None = None
    gimbal_pitch: np.ndarray | None = None
    focal_35mm: float | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    def frame(self) -> LocalFrame:
        return LocalFrame(float(self.lat[0]), float(self.lon[0]), 0.0)

    def enu(self, frame: LocalFrame | None = None) -> np.ndarray:
        frame = frame or self.frame()
        return frame.to_enu(self.lat, self.lon, self.alt)

    def positions_at(
        self, times: np.ndarray, frame: LocalFrame | None = None, offset: float = 0.0
    ) -> np.ndarray:
        """ENU position at each time, linearly interpolated.

        ``offset`` shifts telemetry relative to video (positive = telemetry
        is late), which is what time-sync refinement searches over.
        """
        enu = self.enu(frame)
        tq = np.asarray(times, float) + offset
        return np.stack([np.interp(tq, self.t, enu[:, k]) for k in range(3)], axis=1)

    def speed_at(self, times: np.ndarray, frame: LocalFrame | None = None) -> np.ndarray:
        enu = self.enu(frame)
        if len(self.t) < 2:
            return np.zeros(len(times))
        v = np.gradient(enu, self.t, axis=0)
        sp = np.linalg.norm(v[:, :2], axis=1)
        return np.interp(times, self.t, sp)

    def interp_optional(self, arr: np.ndarray | None, times: np.ndarray) -> np.ndarray | None:
        if arr is None:
            return None
        return np.interp(times, self.t, arr)


_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_KV_RE = re.compile(r"\[\s*([a-z_]+)\s*[:=]\s*([-+0-9.eE/]+)")
_ALT_PAIR_RE = re.compile(r"rel_alt\s*:\s*([-+0-9.]+)")
_OLD_GPS_RE = re.compile(r"GPS\s*\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)")
_OLD_BARO_RE = re.compile(r"(?:BAROMETER|H)\s*[:=]?\s*([-+0-9.]+)\s*m?")
_ANGLE_RE = re.compile(r"(gb_yaw|gb_pitch|yaw|pitch)\s*[:=]\s*([-+0-9.]+)", re.I)


def _secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / (10 ** len(ms))


def parse_srt(path: str | Path) -> Telemetry:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    ts, lats, lons, alts, yaws, pitches = [], [], [], [], [], []
    focal: float | None = None
    for block in blocks:
        tm = _TIME_RE.search(block)
        if not tm:
            continue
        g = tm.groups()
        t = _secs(*g[:4])
        kv = {k.lower(): v for k, v in _KV_RE.findall(block)}
        lat = lon = alt = None
        if "latitude" in kv and "longitude" in kv:
            lat, lon = float(kv["latitude"]), float(kv["longitude"])
            m = _ALT_PAIR_RE.search(block)
            if m:
                alt = float(m.group(1))
            elif "altitude" in kv:
                alt = float(kv["altitude"])
            if "focal_len" in kv and focal is None:
                try:
                    focal = float(kv["focal_len"])
                except ValueError:
                    focal = None
        else:
            m = _OLD_GPS_RE.search(block)
            if m:
                lon, lat = float(m.group(1)), float(m.group(2))
                b = _OLD_BARO_RE.search(block[m.end() :])
                alt = float(b.group(1)) if b else float(m.group(3))
        if lat is None or lon is None or alt is None:
            continue
        if abs(lat) < 1e-7 and abs(lon) < 1e-7:
            continue  # no GPS lock yet
        angles = {k.lower(): float(v) for k, v in _ANGLE_RE.findall(block)}
        ts.append(t)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)
        yaws.append(angles.get("gb_yaw", angles.get("yaw", np.nan)))
        pitches.append(angles.get("gb_pitch", angles.get("pitch", np.nan)))
    if not ts:
        raise ValueError(f"no GPS records found in {path}")
    return _finish(ts, lats, lons, alts, yaws, pitches, focal, f"srt:{Path(path).name}")


def parse_csv(path: str | Path) -> Telemetry:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty telemetry CSV {path}")

    def col(*names: str) -> str | None:
        keys = {k.strip().lower(): k for k in rows[0]}
        for n in names:
            if n in keys:
                return keys[n]
        return None

    ct = col("time", "t", "time_s", "seconds")
    clat = col("lat", "latitude")
    clon = col("lon", "lng", "longitude")
    calt = col("rel_alt", "alt", "altitude", "height", "height_above_takeoff(meters)")
    if not (ct and clat and clon and calt):
        raise ValueError("CSV needs time, lat, lon and alt columns")
    cyaw = col("yaw", "gb_yaw", "heading")
    cpitch = col("gimbal_pitch", "gb_pitch", "pitch")

    def f(row: dict[str, str], c: str | None) -> float:
        if c is None or row.get(c, "") == "":
            return float("nan")
        return float(row[c])

    return _finish(
        [f(r, ct) for r in rows],
        [f(r, clat) for r in rows],
        [f(r, clon) for r in rows],
        [f(r, calt) for r in rows],
        [f(r, cyaw) for r in rows],
        [f(r, cpitch) for r in rows],
        None,
        f"csv:{Path(path).name}",
    )


def load(path: str | Path) -> Telemetry:
    p = Path(path)
    if p.suffix.lower() == ".srt":
        return parse_srt(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return parse_csv(p)
    raise ValueError(f"unsupported telemetry format: {p.suffix}")


def _finish(ts, lats, lons, alts, yaws, pitches, focal, source) -> Telemetry:
    t = np.asarray(ts, float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])  # drop duplicate stamps
    idx = order[keep]
    t = t[keep]

    def arr(xs):
        a = np.asarray(xs, float)[idx]
        return a

    yaw, pitch = arr(yaws), arr(pitches)
    tel = Telemetry(
        t=t,
        lat=arr(lats),
        lon=arr(lons),
        alt=arr(alts),
        yaw=None if np.all(np.isnan(yaw)) else _fill_nan(yaw),
        gimbal_pitch=None if np.all(np.isnan(pitch)) else _fill_nan(pitch),
        focal_35mm=focal,
        source=source,
    )
    _reject_spikes(tel)
    return tel


def _fill_nan(a: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(a)
    if ok.all():
        return a
    x = np.arange(len(a))
    return np.interp(x, x[ok], a[ok])


def _reject_spikes(tel: Telemetry, max_speed: float = 35.0) -> None:
    """Drop fixes implying a physically impossible jump (>35 m/s)."""
    if len(tel) < 3:
        return
    enu = tel.enu()
    keep = np.ones(len(tel), bool)
    last = 0
    for i in range(1, len(tel)):
        dt = max(tel.t[i] - tel.t[last], 1e-3)
        if np.linalg.norm(enu[i] - enu[last]) / dt > max_speed:
            keep[i] = False
        else:
            last = i
    dropped = int((~keep).sum())
    if dropped:
        tel.notes.append(f"dropped {dropped} GPS spikes")
        for name in ("t", "lat", "lon", "alt", "yaw", "gimbal_pitch"):
            v = getattr(tel, name)
            if v is not None:
                setattr(tel, name, v[keep])
