"""Hugging Face Space entry point (Gradio SDK).

The Space runs ``python app.py``. Gradio launches the server, as the Gradio
SDK and ZeroGPU expect, and drone3d's own web app (upload, job queue, 3D
viewer) is served from that same server.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# ZeroGPU hardware refuses to start a Space with no @spaces.GPU function.
# drone3d runs on the CPU, so this placeholder exists only to pass that check;
# off ZeroGPU (locally, CPU Spaces) the `spaces` package is optional.
try:
    import spaces  # noqa: E402

    @spaces.GPU
    def _zerogpu_placeholder() -> bool:
        return True

except ImportError:
    spaces = None

import gradio as gr  # noqa: E402
from gradio.routes import App  # noqa: E402

from drone3d.server import create_app  # noqa: E402

DATA = Path(os.environ.get("DRONE3D_DATA", "/tmp/drone3d_data"))

with gr.Blocks(title="Drone3D") as demo:
    gr.Markdown(
        "## Drone3D\n"
        "Drone video + GPS log → georeferenced, measurable 3D model.\n\n"
        "[Open the app](/) to upload a flight or run the accuracy demo."
    )

# Gradio (and ZeroGPU, which hooks Blocks.launch) must own the server. Hand it
# an app that already holds drone3d's routes: Gradio adds its own routes after
# these, so drone3d's pages win at "/", "/api/*", "/viewer" and "/runs".
server_app = App()
server_app.router.routes.extend(create_app(DATA).router.routes)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860))),
        _app=server_app,
    )
