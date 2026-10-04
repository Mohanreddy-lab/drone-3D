"""Web front-end: upload a flight, watch it process, open the 3D viewer.

``drone3d serve`` -- the full app (needs the ``web`` extra: FastAPI + uvicorn).
``drone3d view RUN`` -- just the viewer on one finished run, standard library only.

Jobs run one at a time on a worker thread: reconstruction already uses every
CPU core, so running two at once would only make both slower.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

VIEWER_DIR = Path(__file__).parent / "viewer"


# ------------------------------------------------------------------ jobs
class JobQueue:
    def __init__(self, data: Path) -> None:
        self.data = data
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        # Jobs interrupted by a restart are marked failed rather than lost.
        for meta in data.glob("*/job.json"):
            j = json.loads(meta.read_text())
            if j["status"] in {"queued", "running"}:
                j["status"] = "failed"
                j["error"] = "server restarted during processing"
                meta.write_text(json.dumps(j))

    def path(self, job_id: str) -> Path:
        return self.data / job_id

    def get(self, job_id: str) -> dict:
        return json.loads((self.path(job_id) / "job.json").read_text())

    def save(self, job: dict) -> None:
        with self.lock:
            (self.path(job["id"]) / "job.json").write_text(json.dumps(job))

    def list(self) -> list[dict]:
        jobs = [json.loads(p.read_text()) for p in self.data.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def create(self, name: str, kind: str, options: dict) -> dict:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path(job_id).mkdir(parents=True)
        job = {
            "id": job_id,
            "name": name,
            "kind": kind,
            "options": options,
            "status": "queued",
            "stage": None,
            "created": time.time(),
            "error": None,
            "log": [],
        }
        self.save(job)
        return job

    def submit(self, job_id: str) -> None:
        self.q.put(job_id)

    def _worker(self) -> None:
        while True:
            job_id = self.q.get()
            try:
                self._run(job_id)
            except Exception:  # noqa: BLE001 - recorded on the job
                job = self.get(job_id)
                job["status"] = "failed"
                job["error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
                self.save(job)

    def _run(self, job_id: str) -> None:
        from drone3d import synth
        from drone3d.pipeline import Options, run

        job = self.get(job_id)
        job["status"] = "running"
        self.save(job)
        root = self.path(job_id)

        def echo(line: str) -> None:
            job["log"] = (job["log"] + [line])[-200:]
            self.save(job)

        def progress(stage: str, frac: float) -> None:
            job["stage"] = stage
            self.save(job)

        inputs = root / "input"
        if job["kind"] == "demo":
            job["stage"] = "rendering synthetic flight"
            self.save(job)
            from drone3d.cli import QUICK

            data = synth.generate(inputs, log=echo, **QUICK)
            video, tel = data["video"], data["telemetry"]
        else:
            video = str(next(inputs.glob("video.*")))
            tels = list(inputs.glob("telemetry.*"))
            tel = str(tels[0]) if tels else None
        opts = Options(**{k: v for k, v in job["options"].items() if v is not None})
        run(video, root / "result", tel, opts, echo=echo, progress=progress)
        if job["kind"] == "demo":
            from drone3d.evaluate import evaluate

            job["evaluation"] = evaluate(root / "result", inputs / "truth.json")
        job["status"] = "done"
        job["stage"] = "done"
        self.save(job)


# ------------------------------------------------------------------ full app
def create_app(data: Path):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    data.mkdir(parents=True, exist_ok=True)
    jobs = JobQueue(data)
    app = FastAPI(title="Drone3D")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (VIEWER_DIR / "app.html").read_text(encoding="utf-8")

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [{k: v for k, v in j.items() if k != "log"} for j in jobs.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        if not (data / job_id / "job.json").exists():
            raise HTTPException(404)
        return jobs.get(job_id)

    @app.post("/api/jobs")
    async def create_job(
        video: UploadFile = File(...),
        telemetry: UploadFile | None = File(None),
        overlap: float = Form(0.85),
        pitch: float = Form(-90.0),
        focal_35mm: float | None = Form(None),
        dense: bool = Form(True),
    ) -> dict:
        opts = {
            "overlap": overlap,
            "default_pitch": pitch,
            "focal_35mm": focal_35mm,
            "dense": dense,
        }
        job = jobs.create(video.filename or "flight", "upload", opts)
        inputs = data / job["id"] / "input"
        inputs.mkdir()
        suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
        with open(inputs / f"video{suffix}", "wb") as fh:
            shutil.copyfileobj(video.file, fh)
        if telemetry is not None and telemetry.filename:
            tsuf = Path(telemetry.filename).suffix.lower() or ".srt"
            with open(inputs / f"telemetry{tsuf}", "wb") as fh:
                shutil.copyfileobj(telemetry.file, fh)
        jobs.submit(job["id"])
        return job

    @app.post("/api/demo")
    def demo() -> dict:
        job = jobs.create("Synthetic test site (known ground truth)", "demo", {})
        jobs.submit(job["id"])
        return job

    @app.get("/viewer")
    def viewer_redirect(job: str) -> RedirectResponse:
        return RedirectResponse(f"/viewer/index.html?scene=/runs/{job}/result/scene.json")

    @app.get("/runs/{job_id}/{path:path}")
    def run_file(job_id: str, path: str):
        target = (data / job_id / path).resolve()
        if not target.is_relative_to(data.resolve()) or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target)

    app.mount("/viewer", StaticFiles(directory=VIEWER_DIR), name="viewer")
    return app


def serve(data: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    print(f"Drone3D running at http://{host}:{port}  (data in {data.resolve()})")
    uvicorn.run(create_app(data), host=host, port=port, log_level="warning")


# ------------------------------------------------------------------ viewer only
class _ViewHandler(http.server.SimpleHTTPRequestHandler):
    run_dir: Path = Path(".")

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/run/"):
            return str(self.run_dir / clean[len("/run/") :])
        if clean in ("/", ""):
            clean = "/index.html"
        return str(VIEWER_DIR / clean.lstrip("/"))

    def log_message(self, *args) -> None:  # quiet
        pass


def view(run_dir: Path, port: int = 8001) -> None:
    if not (run_dir / "scene.json").exists():
        raise SystemExit(f"{run_dir} has no scene.json -- is it a finished run?")
    handler = functools.partial(_ViewHandler)
    _ViewHandler.run_dir = run_dir.resolve()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/index.html?scene=/run/scene.json"
    print(f"viewer: {url}  (Ctrl+C to stop)")
    with contextlib.suppress(Exception):  # headless machines have no browser
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()
