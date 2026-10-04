"""Hugging Face Space entry point (Gradio SDK).

The Space runs ``python app.py``. Gradio launches the server, as the Gradio
SDK and ZeroGPU expect, and drone3d's own web app (upload, job queue, 3D
viewer) is served from that same server.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Spaces turn on Gradio's SSR mode (GRADIO_SSR_MODE=true): a Node server then
# owns the port and renders "/" itself, so drone3d's pages are never reached.
# drone3d needs the plain Python server, so force SSR off before Gradio loads.
os.environ["GRADIO_SSR_MODE"] = "false"

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

# Gradio (and ZeroGPU) must own the server, and a Space may launch `demo`
# through more than one code path. Hook the one place every path goes
# through -- App.create_app -- and put drone3d's routes in front of Gradio's,
# so drone3d serves "/", "/api/*", "/viewer" and "/runs" however it starts.
_DRONE3D_ROUTES = list(create_app(DATA).router.routes)
_gradio_create_app = App.create_app


def _create_app_with_drone3d(*args, **kwargs):
    app = _gradio_create_app(*args, **kwargs)
    if not getattr(app, "_drone3d_routes", False):
        app.router.routes[0:0] = _DRONE3D_ROUTES
        app._drone3d_routes = True
    return app


App.create_app = staticmethod(_create_app_with_drone3d)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860))),
        ssr_mode=False,
    )
