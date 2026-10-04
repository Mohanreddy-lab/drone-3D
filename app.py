"""Hugging Face Space entry point (Gradio SDK).

The Space runs ``python app.py``. It serves drone3d's own web app (upload,
job queue, 3D viewer) and mounts a small Gradio page at /gradio, which the
Gradio SDK expects to find.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from drone3d.server import create_app  # noqa: E402

DATA = Path(os.environ.get("DRONE3D_DATA", "/tmp/drone3d_data"))

app = create_app(DATA)

with gr.Blocks(title="Drone3D") as about:
    gr.Markdown(
        "## Drone3D\n"
        "Drone video + GPS log → georeferenced, measurable 3D model.\n\n"
        "[Open the app](/) to upload a flight or run the accuracy demo."
    )

app = gr.mount_gradio_app(app, about, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
