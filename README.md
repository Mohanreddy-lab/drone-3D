# drone3d: drone video to a measurable 3D model in one pass

> Input: a drone video and its flight log (DJI `.SRT` or CSV).
> Output: a georeferenced 3D model you can measure in the browser, plus LAS, DSM, orthophoto and an accuracy report.
> No manual tie points, no hand-picked photos, no GPU required.

This is a working prototype. Everything below was measured by running it.

```
drone video ─┐
             ├─► 1 sync ─► 2 keyframes ─► 3 match ─► 4 SfM ─► 5 georef ─► 6 dense ─► 7 export ─► viewer
flight log ──┘   clock      sharpest,      GPS-       COLMAP    7-DoF +     CPU SGBM    LAS/DSM/   measure
                 offset     by overlap     guided     + GPS     gimbal      or CUDA     ortho/     distance,
                 search                    pairs      prior BA  levelling   PatchMatch  GeoJSON    area, volume
```

## Why it cuts manual work

| Usual manual step | What drone3d does instead |
|---|---|
| Extract frames from video by hand, delete blurry ones | **Overlap-driven keyframe selection.** It keeps a frame each time the drone has moved `(1 − overlap) × footprint` metres (from GPS, altitude, FOV and gimbal pitch), and picks the **sharpest** frame in each window. Motion-blurred frames are never chosen. |
| Align the telemetry clock with the video | **Automatic clock sync.** It searches ±1 s for the offset that best fits the solved camera path to GPS. |
| Wait for exhaustive matching (O(n²)) | **GPS-guided pairing.** Each frame is matched only against sequence neighbours and frames within a GPS radius. Adjacent survey lines still connect. |
| Place ground control points to stop the model tilting | **Gimbal levelling.** A stabilised gimbal holds the horizon and logs its pitch, so each camera knows where "down" is. That removes the tilt consumer-GPS heights put into narrow sites (2.3° in our test, see below). |
| Buy a GPU for dense matching | **CPU fallback.** Pairwise semi-global matching on solved poses, fused by multi-view voting in a voxel grid. Uses COLMAP PatchMatch automatically when CUDA is present. |
| Export to separate GIS and measuring tools | **One run directory** with LAS (UTM, CRS embedded), DSM GeoTIFF, orthophoto and camera GeoJSON (all with world files and `.prj`), plus a browser viewer with measurement tools. |
| Guess how accurate a measurement is | **Error bars on every measurement.** Reprojection error × GSD gives in-model precision. A bootstrap of the GPS fit gives the scale uncertainty. Absolute accuracy is reported as GPS-limited. |

## Measured results (synthetic site, known ground truth)

`drone3d demo` renders a test site with exactly known dimensions: two buildings and a square-pyramid stockpile. It flies a 3-line survey over the site at 40 m with the gimbal at −75°, encodes the result as H.264 like a real drone, and writes a DJI-style SRT. The SRT's GPS has consumer-grade error (1.5 m horizontal / 2.5 m vertical, slowly wandering), and 6% of frames are motion-blurred. The pipeline is then scored against the truth. Runs below were on a 4-core CPU, no GPU.

| Stage | Result |
|---|---|
| Keyframe selection | 759 video frames → **95 keyframes** (8× fewer), blurred frames skipped |
| Structure from Motion | **95 / 95** registered, mean reprojection error **0.22 px** |
| Georeferencing | auto clock offset +0.04 s; gimbal levelling corrected **2.27°** of GPS-induced tilt |
| Dense cloud | 647 k points (CPU semi-global matching, 102 stereo pairs) |
| Total time | **~3 min** from video to finished outputs (SfM ~2.5 min, dense ~20 s) |

| Measurement taken in the model | True | Measured | Error |
|---|---:|---:|---:|
| Building A height | 8.00 m | 8.03 m | +0.3% |
| Building B height | 12.00 m | 12.04 m | +0.3% |
| Building A footprint | 20.0 × 12.0 m | 20.6 × 12.4 m | +3% |
| Building B footprint | 15.0 × 15.0 m | 15.4 × 15.4 m | +2.7% |
| Stockpile volume | 512.0 m³ | 522.7 m³ | **+2.1%** |

| Shape vs position | Value |
|---|---|
| Surface shape error (RMSE vs true surface) | **0.13 m**, > 99% of points within 25 cm |
| …the same before gimbal levelling | 14% of points within 25 cm (the model was tilted 2.3°) |
| Whole-model offset on the globe (absolute) | 1.99 m horizontal, −1.37 m vertical |

The `--quick` demo (640×360, 2 survey lines, about 75 s on a 4-core CPU) gives:
- stockpile volume 503 m³ (−1.7%)
- heights within 2%
- 98% of points within 25 cm of the true shape

**How to read these numbers:**
- **Inside the model** (lengths, heights, areas, volumes), accuracy is centimetres to a few percent.
- **On the globe**, the whole model is offset by about the drone's GPS error: about 2 m here, as simulated. Survey-grade absolute positions need an RTK/PPK drone or ground control points. The report and viewer state this rather than hiding it.
- The evaluation removes that offset before measuring, the same way a surveyor draws a stockpile outline around the pile they see in the model.
- Footprints read 2–3% large because wall points at the roof edge fall into DSM cells just outside the building.

## Quick start

```bash
cd drone3d
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[web,dev]"

drone3d demo --quick -o demo            # ~75 s: render, reconstruct, score
drone3d view demo/result                # opens the 3D viewer
```

On your own footage:

```bash
drone3d run DJI_0001.MP4 -t DJI_0001.SRT -o survey1
drone3d view survey1
```

The web app (upload form, job queue, viewer, built-in demo button):

```bash
drone3d serve --port 8000               # http://127.0.0.1:8000
```

### Inputs

| | Supported |
|---|---|
| Video | anything OpenCV/FFmpeg decodes (MP4/MOV, H.264/H.265), any resolution (keyframes are downscaled to `--max-dim`, default 1600 px) |
| Telemetry | DJI `.SRT` (modern `[latitude: …]` style and older `GPS(…) BAROMETER` style); CSV with `time, lat, lon, alt` (+ optional `yaw`, `gimbal_pitch`) for Airdata / Mission Planner / PX4 exports |
| No telemetry | still works: keyframes are chosen by optical flow and the model is unscaled |

Useful flags: `--overlap 0.85`, `--pitch -90` (gimbal pitch if not logged), `--focal-35mm 24`, `--time-offset`, `--no-dense`, `--resume`. Run `drone3d run -h` for all of them.

### Outputs (`<out>/`)

| File | What it is |
|---|---|
| `cloud.las` | dense point cloud, UTM coordinates, CRS embedded (QGIS, CloudCompare, ArcGIS) |
| `cloud.ply`, `sparse.ply` | same cloud in local ENU metres (MeshLab, Blender, the viewer) |
| `dsm.tif` + `.tfw` + `.prj` | digital surface model, float32 metres |
| `ortho.png` + `.pgw` + `.prj` | true orthophoto (top-down colour of the surface) |
| `dsm_hillshade.png` | quick-look elevation render |
| `cameras.geojson` | solved flight path and keyframe positions (WGS84) |
| `report.json` | every stage's statistics, accuracy summary, timings |
| `scene.json` | what the viewer loads |
| `images/`, `frames.csv`, `work/` | keyframes, their GPS, the COLMAP database and model |

### Viewer tools

- **Point:** latitude/longitude and height of any point.
- **Distance:** 3D, horizontal and vertical distance, with an uncertainty.
- **Height:** vertical difference between two points.
- **Area:** planimetric polygon area and perimeter.
- **Volume:** cut/fill above a plane fitted through the outline, the standard stockpile method; it works on sloping ground.
- **Export:** CSV or GeoJSON of all measurements.

three.js is vendored (MIT), so the viewer works offline.

## Code map

| Module | Role |
|---|---|
| `telemetry.py` | SRT/CSV parsing, GPS spike rejection, interpolation to frame times |
| `frames.py` | sharpness + optical flow analysis, overlap-driven keyframe choice |
| `sfm.py` | COLMAP features, GPS-guided matching, incremental mapping with GPS-prior BA |
| `georef.py` | robust Helmert fit, clock-offset search, gimbal levelling, accuracy model |
| `dense.py` | CPU SGBM stereo + voxel fusion; CUDA PatchMatch when available |
| `export.py` | PLY / LAS / GeoTIFF / orthophoto / GeoJSON |
| `measure.py` | distance, area, volume (reference implementation of the viewer's maths) |
| `synth.py`, `evaluate.py` | synthetic survey generator and ground-truth scoring |
| `pipeline.py`, `cli.py`, `server.py`, `viewer/` | orchestration, CLI, web app, 3D viewer |

## Tests

```bash
pytest                      # unit tests, ~1 s
DRONE3D_E2E=1 pytest        # adds a full synthetic run, ~3 min
ruff check .
```

## Limitations and next steps

- **Absolute accuracy:** GPS-limited without RTK/PPK or GCPs. Next steps: a GCP picker in the viewer, and RTK flags read from the SRT.
- **Output form:** the model is a point cloud plus DSM. Next steps: textured mesh (Poisson + texturing) and 3D Tiles for very large sites.
- **Rolling shutter:** not modelled yet. Fast flights with CMOS rolling shutter will lose a little precision.
- **Long flights (thousands of keyframes):** should use COLMAP's global/hierarchical mapper, which is already in pycolmap.
- **Real footage:** the synthetic benchmark proves the geometry, but real flights add exposure changes, moving objects and vegetation. Validating on public datasets (for example the OpenDroneMap samples) is the next milestone.

## Relationship to `home-core`

`home-core` is the stdlib-only ports package for HOME AI and must not import anything heavy. Its `test_import_weight.py` enforces that. `drone3d` is therefore a separate package with its own `pyproject.toml`; it lives here only while it's a prototype. Drone imagery of a property, together with its GPS track, is location data about a household. If drone3d is integrated later, its stores and any network calls should go through `home_core`'s `Classified`/`EgressRequest` types, and processing should stay local by default. It runs entirely offline today.
