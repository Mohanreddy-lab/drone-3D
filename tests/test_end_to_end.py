"""Full pipeline on a small synthetic flight (about 2 minutes on 4 cores).

Opt in with ``DRONE3D_E2E=1 pytest``.
"""

import json
import os

import pytest

from drone3d import synth
from drone3d.cli import QUICK
from drone3d.evaluate import evaluate
from drone3d.pipeline import Options, run

pytestmark = pytest.mark.skipif(not os.environ.get("DRONE3D_E2E"), reason="set DRONE3D_E2E=1")


def test_synthetic_site_is_measured_accurately(tmp_path):
    data = synth.generate(tmp_path / "in", log=lambda *_: None, **QUICK)
    report = run(data["video"], tmp_path / "out", data["telemetry"], Options(), echo=None)
    assert report["sfm"]["images_registered"] >= 0.9 * report["sfm"]["images_input"]
    assert report["sfm"]["mean_reprojection_error_px"] < 1.0
    ev = evaluate(tmp_path / "out", data["truth"])
    a = next(o for o in ev["objects"] if o["name"] == "building_A")
    assert a["measured"]["height_m"] == pytest.approx(8.0, rel=0.08)
    assert a["measured"]["length_x_m"] == pytest.approx(20.0, rel=0.08)
    pile = next(o for o in ev["objects"] if o["name"] == "stockpile")
    assert pile["measured"]["volume_m3"] == pytest.approx(512.0, rel=0.12)
    for name in ("cloud.las", "dsm.tif", "ortho.png", "cameras.geojson", "scene.json"):
        assert (tmp_path / "out" / name).exists()
    assert json.loads((tmp_path / "out" / "scene.json").read_text())["georeferenced"]
