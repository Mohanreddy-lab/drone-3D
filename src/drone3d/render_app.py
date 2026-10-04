"""Render entry point for the Drone3D FastAPI application."""

from __future__ import annotations

import os
from pathlib import Path

from drone3d.server import create_app

app = create_app(Path(os.environ.get("DRONE3D_DATA", "/var/data")))