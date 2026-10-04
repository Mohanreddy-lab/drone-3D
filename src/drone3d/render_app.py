"""Render entry point for the Drone3D FastAPI application."""

from __future__ import annotations

import os
from pathlib import Path

from drone3d.server import create_app



def _data_path() -> Path:
	configured = Path(os.environ.get("DRONE3D_DATA", "/var/data"))
	try:
		configured.mkdir(parents=True, exist_ok=True)
	except PermissionError:
		configured = Path("/tmp/drone3d-data")
		configured.mkdir(parents=True, exist_ok=True)
	return configured


app = create_app(_data_path())