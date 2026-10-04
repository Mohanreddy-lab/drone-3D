"""Dense reconstruction that runs on a plain CPU.

COLMAP's PatchMatch MVS needs CUDA. When a GPU is present we use it; when not
(a laptop, a judge's machine, a CI runner) we fall back to pairwise
semi-global matching:

1. for every registered image pick 1-2 partners with a good baseline-to-depth
   ratio and many shared sparse points;
2. stereo-rectify the pair with the solved poses, run OpenCV SGBM, and turn
   disparity into metric depth;
3. fuse all depth maps in a voxel grid and keep only voxels seen
   consistently from at least ``min_views`` depth maps -- the vote is what
   removes the speckle a single stereo pair produces.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def densify(
    rec: pycolmap.Reconstruction,
    image_dir: Path,
    work_dir: Path,
    *,
    max_dim: int = 1000,
    voxel: float = 0.2,
    min_views: int = 2,
    neighbours: int = 2,
    target_ratio: float = 0.18,
    stride: int = 2,
    prefer_gpu: bool = True,
    log: Callable = print,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if prefer_gpu and pycolmap.has_cuda:
        try:
            return _densify_patchmatch(rec, image_dir, work_dir, log)
        except Exception as exc:  # pragma: no cover - GPU path
            log(f"PatchMatch failed ({exc}); falling back to CPU stereo")
    return _densify_sgbm(
        rec, image_dir, max_dim, voxel, min_views, neighbours, target_ratio, stride, log
    )


def _camera_K(cam: pycolmap.Camera, scale: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(cam.params, float)
    model = cam.model_name if hasattr(cam, "model_name") else str(cam.model)
    if "SIMPLE_RADIAL" in model:
        f, cx, cy, k = p
        fx = fy = f
        dist = np.array([k, 0, 0, 0], float)
    elif "SIMPLE_PINHOLE" in model:
        f, cx, cy = p
        fx = fy = f
        dist = np.zeros(4)
    elif "PINHOLE" in model:
        fx, fy, cx, cy = p
        dist = np.zeros(4)
    elif "RADIAL" in model:
        f, cx, cy, k1, k2 = p
        fx = fy = f
        dist = np.array([k1, k2, 0, 0], float)
    else:  # OPENCV and friends
        fx, fy, cx, cy = p[:4]
        dist = np.array(list(p[4:8]) + [0] * (4 - len(p[4:8])), float)
    K = np.array([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]])
    return K, dist


def _pose(img: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(img.cam_from_world().matrix())
    return m[:, :3], m[:, 3]


def _pick_pairs(rec, neighbours: int, target_ratio: float) -> tuple[list[tuple[int, int]], dict]:
    ids = [i for i in rec.reg_image_ids()]
    centers = {i: np.asarray(rec.images[i].projection_center()) for i in ids}
    vis: dict[int, set[int]] = {}
    depth: dict[int, float] = {}
    for i in ids:
        img = rec.images[i]
        R, t = _pose(img)
        pids = [p.point3D_id for p in img.points2D if p.has_point3D()]
        vis[i] = set(pids)
        if pids:
            xyz = np.array([rec.points3D[p].xyz for p in pids])
            z = (xyz @ R.T + t)[:, 2]
            depth[i] = float(np.median(z[z > 0])) if (z > 0).any() else 0.0
        else:
            depth[i] = 0.0
    pairs: list[tuple[int, int]] = []
    seen = set()
    for i in ids:
        if depth[i] <= 0 or len(vis[i]) < 30:
            continue
        scored = []
        for j in ids:
            if j == i:
                continue
            shared = len(vis[i] & vis[j])
            if shared < 30:
                continue
            ratio = np.linalg.norm(centers[i] - centers[j]) / depth[i]
            if ratio < 0.04 or ratio > 0.35:
                continue
            score = shared * np.exp(-((np.log(ratio / target_ratio)) ** 2) / 0.5)
            scored.append((score, j))
        scored.sort(reverse=True)
        for _, j in scored[:neighbours]:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((i, j))
    return pairs, depth


def _depth_range(rec, img_id: int) -> tuple[float, float]:
    img = rec.images[img_id]
    R, t = _pose(img)
    pids = [p.point3D_id for p in img.points2D if p.has_point3D()]
    xyz = np.array([rec.points3D[p].xyz for p in pids])
    z = (xyz @ R.T + t)[:, 2]
    z = z[z > 0]
    lo, hi = np.percentile(z, [0.5, 99.5])
    return float(lo), float(hi)


def _stereo_pair(rec, i: int, j: int, images: dict, K, dist, size, stride: int):
    """Depth points (world) and colours for reference image ``i``."""
    Ri, ti = _pose(rec.images[i])
    Rj, tj = _pose(rec.images[j])
    R = Rj @ Ri.T
    T = tj - R @ ti
    R1, R2, P1, P2, _Q, _, _ = cv2.stereoRectify(
        K,
        dist.reshape(1, -1),
        K,
        dist.reshape(1, -1),
        size,
        R,
        T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    shift = P2[1, 3] if vertical else P2[0, 3]
    if shift > 0:
        return None  # the mirrored pair (j, i) is the one SGBM can process
    f = P1[0, 0]
    baseline = abs(shift) / f
    m1 = cv2.initUndistortRectifyMap(K, dist.reshape(1, -1), R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K, dist.reshape(1, -1), R2, P2, size, cv2.CV_32FC1)
    left = cv2.remap(images[i], *m1, cv2.INTER_LINEAR)
    right = cv2.remap(images[j], *m2, cv2.INTER_LINEAR)
    valid = cv2.remap(np.full(size[::-1], 255, np.uint8), *m1, cv2.INTER_NEAREST) > 0
    zlo, zhi = _depth_range(rec, i)
    dmin = f * baseline / (zhi * 1.4)
    dmax = f * baseline / (zlo * 0.6)
    min_disp = int(np.floor(dmin))
    num = int(np.ceil((dmax - min_disp) / 16.0) * 16)
    num = max(16, min(num, 256))
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    if vertical:
        gl, gr = np.ascontiguousarray(gl.T), np.ascontiguousarray(gr.T)
    if min_disp + num > 0.8 * gl.shape[1]:
        return None  # baseline too wide for this image size: overlap too small
    # SGBM leaves the leftmost (minDisparity + numDisparities) columns empty.
    # Padding first keeps that band -- on a short image axis it is up to half
    # the depth map.
    pad = min_disp + num
    gl = cv2.copyMakeBorder(gl, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    gr = cv2.copyMakeBorder(gr, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    block = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=12,
        speckleWindowSize=120,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(gl, gr).astype(np.float32)[:, pad:] / 16.0
    if vertical:
        disp = disp.T
    ok = (disp > min_disp) & valid
    mask = np.zeros_like(ok)
    mask[::stride, ::stride] = True
    ok &= mask
    v, u = np.nonzero(ok)
    d = disp[v, u]
    Z = f * baseline / d
    keep = (zlo * 0.6 < Z) & (zhi * 1.4 > Z)
    u, v, Z = u[keep], v[keep], Z[keep]
    cx, cy = P1[0, 2], P1[1, 2]
    X = np.stack([(u - cx) * Z / f, (v - cy) * Z / f, Z], 1)
    Xc = X @ R1  # R1^T applied to row vectors: rectified -> camera i
    Xw = (Xc - ti) @ Ri  # Ri^T (x - t)
    colors = left[v, u][:, ::-1]  # BGR -> RGB
    return Xw.astype(np.float32), colors


def _densify_sgbm(rec, image_dir, max_dim, voxel, min_views, neighbours, target_ratio, stride, log):
    cam = rec.cameras[next(iter(rec.cameras))]
    scale = min(1.0, max_dim / max(cam.width, cam.height))
    K, dist = _camera_K(cam, scale)
    size = (int(round(cam.width * scale)), int(round(cam.height * scale)))
    pairs, _ = _pick_pairs(rec, neighbours, target_ratio)
    log(f"dense: CPU semi-global matching on {len(pairs)} stereo pairs at {size[0]}x{size[1]}")
    cache: dict[int, np.ndarray] = {}

    def load(i: int) -> np.ndarray:
        if i not in cache:
            im = cv2.imread(str(Path(image_dir) / rec.images[i].name))
            if scale < 1.0:
                im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            cache[i] = im
            if len(cache) > 12:
                cache.pop(next(iter(cache)))
        return cache[i]

    sparse = np.array([p.xyz for p in rec.points3D.values()])
    lo = np.percentile(sparse, 0.5, axis=0) - 5
    hi = np.percentile(sparse, 99.5, axis=0) + 5

    keys_all, xyz_all, rgb_all = [], [], []
    done = 0
    for a, b in pairs:
        out = None
        for i, j in ((a, b), (b, a)):
            imgs = {i: load(i), j: load(j)}
            out = _stereo_pair(rec, i, j, imgs, K, dist, size, stride)
            if out is not None:
                break
        if out is None:
            continue
        pts, cols = out
        inb = np.all((pts > lo) & (pts < hi), axis=1)
        pts, cols = pts[inb], cols[inb]
        if len(pts) == 0:
            continue
        key = _voxel_keys(pts, voxel)
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float32)
        mean = np.stack([np.bincount(inv, pts[:, k]) for k in range(3)], 1) / cnt[:, None]
        col = np.stack([np.bincount(inv, cols[:, k]) for k in range(3)], 1) / cnt[:, None]
        keys_all.append(uk)
        xyz_all.append(mean.astype(np.float32))
        rgb_all.append(col.astype(np.float32))
        done += 1
        if done % 20 == 0:
            log(f"  {done}/{len(pairs)} pairs")
    if not keys_all:
        raise RuntimeError("dense matching produced no points")
    keys = np.concatenate(keys_all)
    xyz = np.concatenate(xyz_all)
    rgb = np.concatenate(rgb_all)
    uk, inv, views = np.unique(keys, return_inverse=True, return_counts=True)
    sx = np.stack([np.bincount(inv, xyz[:, k]) for k in range(3)], 1) / views[:, None]
    sc = np.stack([np.bincount(inv, rgb[:, k]) for k in range(3)], 1) / views[:, None]
    keep = views >= min_views
    pts = sx[keep].astype(np.float32)
    colors = np.clip(sc[keep], 0, 255).astype(np.uint8)
    pts, colors = _remove_outliers(pts, colors, voxel)
    stats = {
        "method": "cpu-sgbm",
        "pairs": len(pairs),
        "depth_maps": done,
        "voxel_m": voxel,
        "min_views": min_views,
        "points": int(len(pts)),
    }
    log(f"dense: fused {len(pts):,} points (voxel {voxel} m, >= {min_views} views)")
    return pts, colors, stats


def _voxel_keys(pts: np.ndarray, voxel: float) -> np.ndarray:
    q = np.floor(pts / voxel).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def _remove_outliers(pts: np.ndarray, colors: np.ndarray, voxel: float):
    """Drop isolated voxels: keep points with >= 4 occupied neighbours."""
    if len(pts) < 100:
        return pts, colors
    q = np.floor(pts / voxel).astype(np.int64)
    base = _pack(q)
    occupied = np.sort(base)
    count = np.zeros(len(pts), np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                k = _pack(q + np.array([dx, dy, dz]))
                pos = np.searchsorted(occupied, k)
                pos = np.minimum(pos, len(occupied) - 1)
                count += occupied[pos] == k
    keep = count >= 4
    return pts[keep], colors[keep]


def _pack(q: np.ndarray) -> np.ndarray:
    q = q + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def _densify_patchmatch(rec, image_dir, work_dir, log):  # pragma: no cover - needs CUDA
    mvs = Path(work_dir) / "mvs"
    if mvs.exists():
        shutil.rmtree(mvs)
    mvs.mkdir(parents=True)
    sparse = mvs / "sparse_in"
    sparse.mkdir()
    rec.write(sparse)
    log("dense: undistorting images for CUDA PatchMatch")
    pycolmap.undistort_images(mvs, sparse, image_dir)
    log("dense: PatchMatch stereo (GPU)")
    pycolmap.patch_match_stereo(mvs)
    ply = mvs / "fused.ply"
    pycolmap.stereo_fusion(ply, mvs)
    from drone3d.export import read_ply

    pts, cols = read_ply(ply)
    return pts, cols, {"method": "cuda-patchmatch", "points": int(len(pts))}
