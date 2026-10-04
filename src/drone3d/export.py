"""Deliverables: point clouds, rasters, camera track, and the viewer scene.

Coordinates
-----------
* ``*.ply`` and the web viewer use local ENU metres (float32-safe).
* ``cloud.las`` is in the UTM zone of the site (EPSG 326xx / 327xx) with the
  CRS embedded, so it drops straight into QGIS, CloudCompare or ArcGIS.
* ``dsm.tif`` / ``ortho.png`` carry ``.tfw``/``.pgw`` world files plus a
  ``.prj`` in the same UTM CRS.
* ``cameras.geojson`` is WGS84 lon/lat.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from drone3d.geo import LocalFrame


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes())


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal binary/ASCII PLY vertex reader (xyz + optional rgb)."""
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        body = fh.read()
    n = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    fmt = "binary_little_endian"
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            props.append((parts[-1], parts[1]))
    tmap = {
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "uchar": "u1",
        "uint8": "u1",
        "int": "i4",
        "uint": "u4",
        "short": "i2",
        "ushort": "u2",
    }
    if fmt == "ascii":
        data = np.loadtxt(body.decode().splitlines()[:n])
        names = [p[0] for p in props]
        xyz = data[:, [names.index(c) for c in "xyz"]].astype(np.float32)
        if "red" in names:
            rgb = data[:, [names.index(c) for c in ("red", "green", "blue")]].astype(np.uint8)
        else:
            rgb = np.full((n, 3), 200, np.uint8)
        return xyz, rgb
    end = "<" if "little" in fmt else ">"
    dt = np.dtype([(name, end + tmap[t]) for name, t in props])
    arr = np.frombuffer(body[: n * dt.itemsize], dt, count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
    if "red" in arr.dtype.names:
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 200, np.uint8)
    return xyz, rgb


def enu_to_utm(frame: LocalFrame, pts: np.ndarray) -> tuple[np.ndarray, int]:
    from pyproj import Transformer

    epsg = frame.utm_epsg()
    lat, lon, h = frame.to_geodetic(pts.astype(np.float64))
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e, n = tr.transform(lon, lat)
    return np.stack([e, n, h], 1), epsg


def write_las(path: Path, frame: LocalFrame, pts: np.ndarray, rgb: np.ndarray) -> int:
    import laspy
    from pyproj import CRS

    utm, epsg = enu_to_utm(frame, pts)
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = np.floor(utm.min(axis=0))
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_crs(CRS.from_epsg(epsg))
    las = laspy.LasData(header)
    las.x, las.y, las.z = utm[:, 0], utm[:, 1], utm[:, 2]
    las.red, las.green, las.blue = (rgb[:, k].astype(np.uint16) * 257 for k in range(3))
    las.write(str(path))
    return epsg


def write_cameras_geojson(path: Path, frame: LocalFrame, names, centers: np.ndarray, gps) -> None:
    lat, lon, h = frame.to_geodetic(centers)
    feats = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(a), float(b)] for a, b in zip(lon, lat, strict=True)],
            },
            "properties": {"name": "solved flight path"},
        }
    ]
    for k, name in enumerate(names):
        props = {"name": name, "height_m": float(h[k])}
        if gps is not None:
            props["gps_offset_m"] = float(np.linalg.norm(gps[k] - centers[k]))
        feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon[k]), float(lat[k]), float(h[k])],
                },
                "properties": props,
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


def rasterize(pts: np.ndarray, rgb: np.ndarray, gsd: float):
    """Top-down DSM (max z per cell) and true-orthophoto (colour of that z).

    Returns (dsm[float32, nan=empty], ortho[uint8 BGR], x0, y_top).
    """
    x0, y0 = np.floor(pts[:, :2].min(axis=0) / gsd) * gsd
    x1, y1 = np.ceil(pts[:, :2].max(axis=0) / gsd) * gsd
    w = int(round((x1 - x0) / gsd)) + 1
    h = int(round((y1 - y0) / gsd)) + 1
    col = ((pts[:, 0] - x0) / gsd).astype(np.int64)
    row = ((y1 - pts[:, 1]) / gsd).astype(np.int64)  # north up
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)
    cell = row * w + col
    order = np.lexsort((pts[:, 2], cell))  # by cell, then z ascending
    last = np.r_[cell[order][1:] != cell[order][:-1], True]
    top = order[last]
    dsm = np.full(h * w, np.nan, np.float32)
    dsm[cell[top]] = pts[top, 2]
    ortho = np.zeros((h * w, 3), np.uint8)
    ortho[cell[top]] = rgb[top][:, ::-1]
    dsm = dsm.reshape(h, w)
    ortho = ortho.reshape(h, w, 3)
    dsm, ortho = _fill_holes(dsm, ortho.astype(np.float32))
    return dsm, ortho, float(x0), float(y1)


def _fill_holes(dsm: np.ndarray, ortho: np.ndarray, iterations: int = 3):
    """Fill isolated empty cells from their 8-neighbourhood mean."""
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    for _ in range(iterations):
        empty = np.isnan(dsm)
        if not empty.any():
            break
        valid = (~empty).astype(np.float32)
        n = cv2.filter2D(valid, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        zsum = cv2.filter2D(np.nan_to_num(dsm), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        csum = cv2.filter2D(ortho * valid[..., None], -1, kernel, borderType=cv2.BORDER_CONSTANT)
        fill = empty & (n >= 3)
        dsm = np.where(fill, zsum / np.maximum(n, 1), dsm)
        ortho = np.where(fill[..., None], csum / np.maximum(n, 1)[..., None], ortho)
    return dsm.astype(np.float32), np.clip(ortho, 0, 255).astype(np.uint8)


def hillshade(dsm: np.ndarray, gsd: float) -> np.ndarray:
    z = np.nan_to_num(dsm, nan=np.nanmin(dsm) if np.isfinite(dsm).any() else 0.0)
    gy, gx = np.gradient(z, gsd)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    zn = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    color = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).astype(np.float32)
    out = color * (0.35 + 0.65 * np.clip(shade, 0, 1))[..., None]
    out[np.isnan(dsm)] = 0
    return out.astype(np.uint8)


def write_rasters(
    out: Path, frame: LocalFrame, pts: np.ndarray, rgb: np.ndarray, gsd: float
) -> dict:
    from pyproj import CRS, Transformer

    dsm, ortho, x0, y_top = rasterize(pts, rgb, gsd)
    epsg = frame.utm_epsg()
    # World file anchors the raster in UTM. ENU and UTM differ by the grid
    # convergence angle (well under a degree near the origin) -- acceptable
    # for a site-sized prototype, and documented in the report.
    lat, lon, _ = frame.to_geodetic(np.array([x0, y_top, 0.0]))
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e0, n0 = tr.transform(float(lon), float(lat))
    world = f"{gsd}\n0.0\n0.0\n{-gsd}\n{e0 + gsd / 2}\n{n0 - gsd / 2}\n"
    wkt = CRS.from_epsg(epsg).to_wkt("WKT1_ESRI")
    cv2.imwrite(str(out / "dsm.tif"), dsm.astype(np.float32))
    (out / "dsm.tfw").write_text(world)
    (out / "dsm.prj").write_text(wkt)
    cv2.imwrite(str(out / "ortho.png"), ortho)
    (out / "ortho.pgw").write_text(world)
    (out / "ortho.prj").write_text(wkt)
    cv2.imwrite(str(out / "dsm_hillshade.png"), hillshade(dsm, gsd))
    return {
        "gsd_m": gsd,
        "width": dsm.shape[1],
        "height": dsm.shape[0],
        "epsg": epsg,
        "origin_utm": [e0, n0],
    }
