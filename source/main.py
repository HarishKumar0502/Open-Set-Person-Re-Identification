from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import logging
import re
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from source.config import PROJECT_ROOT
from source.dependencies import S, VIDEO_JOBS, init_state, is_initialised
from source.errors import DetectorUnavailableError, PS1Error
from source.logic.video.video_tracking import VIDEO_EXTENSIONS
from source.schemas.requests import video_job_params
from source.utils.logging_config import setup_logging
from worker import OUTPUT_STEM, Job

log = logging.getLogger("ps1.api")
ROOT = PROJECT_ROOT
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
CHUNK = 1024 * 1024
MEDIA_TYPES = {".mp4": "video/mp4", ".csv": "text/csv", ".json": "application/json"}


def _is_benign_disconnect(context: dict) -> bool:
    exc = context.get("exception")
    return (isinstance(exc, (ConnectionResetError, ConnectionAbortedError))
            and "_call_connection_lost" in repr(context.get("handle") or context.get("message", "")))


def _install_loop_guards(loop: asyncio.AbstractEventLoop):
    if sys.platform != "win32" or not isinstance(loop, getattr(asyncio, "ProactorEventLoop", ())):
        return
    log.warning("Running on the Windows Proactor event loop: a client that aborts a connection during accept "
                "can make it stop accepting connections. Start the server with 'python -m source.main'.")
    previous = loop.get_exception_handler()

    def handler(lp, context):
        if _is_benign_disconnect(context):
            log.debug("Client disconnected abruptly: %s", context.get("exception"))
            return
        if previous is not None:
            previous(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(handler)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _install_loop_guards(asyncio.get_running_loop())
    if not is_initialised():
        init_state()
    VIDEO_JOBS.sweep_stale_inputs()
    yield
    VIDEO_JOBS.shutdown()


app = FastAPI(title="PS1 Open-Set Person ReID", lifespan=lifespan,
              description="Upload a video to `POST /api/video/jobs`, poll `GET /api/video/jobs/{id}`, then download "
                          "its files. Every person track is labelled with an enrolled identity or UNKNOWN. The "
                          "inference backend (TensorRT / ONNX Runtime / PyTorch, GPU or CPU) is chosen internally.")
app.mount("/static", StaticFiles(directory=ROOT / "webapp" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "webapp" / "templates")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


def _settings() -> dict:
    e = S.engine
    g = e.gallery
    return {"threshold": e.matcher.threshold, "margin": e.matcher.margin,
            "aggregation": S.cfg["gallery"]["aggregation"],
            "gallery_images": len(g), "gallery_identities": int(len(g.identities)),
            "device": f"{e.reid.name} ({e.reid.device})", "gallery_source": Path(g.source).name}


def _video_defaults() -> dict:
    v = S.cfg.get("video_tracking", {}) or {}
    ident = v.get("identity", {}) or {}
    return {"det_confidence": v.get("det_confidence", 0.25), "frame_stride": v.get("frame_stride", 1),
            "reid_interval": ident.get("reid_interval", 3), "min_votes": ident.get("min_votes", 2),
            "imgsz": v.get("imgsz", 640), "max_upload_mb": S.cfg.get("webapp", {}).get("max_video_upload_mb", 1024),
            "extensions": ",".join(sorted(VIDEO_EXTENSIONS))}


def _job_or_404(job_id: str) -> Job:
    job = VIDEO_JOBS.get(job_id) if JOB_ID.match(job_id) else None
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"s": _settings(), "v": _video_defaults()})


@app.get("/api/health", tags=["system"])
def health():
    from source.logic.model_manager import available_backends
    from source.models.tensorrt.gpu_runtime import probe_cuda, tensorrt_version

    e = S.engine
    gpu = probe_cuda()
    st = e.settings
    return {
        "status": "ok", **_settings(),
        "reid": {"backend": e.reid.name, "device": e.reid.device, "info": e.backend_info,
                 "checkpoint": Path(st.checkpoint).name, "embedding_dim": st.embedding_dim,
                 "input_size_hw": list(st.input_size)},
        "gallery_consistency": e.consistency,
        "hardware": {"cuda_gpu": gpu.get("name") if gpu["available"] else None,
                     "cuda_note": None if gpu["available"] else gpu.get("reason"),
                     "tensorrt_version": tensorrt_version()},
        "installed_inference_backends": available_backends(st),
        "artifacts": {"checkpoint_present": Path(st.checkpoint).is_file(),
                      "onnx_model_present": st.onnx_path.is_file(),
                      "tensorrt_engine_present": st.tensorrt_path.is_file()},
        "jobs_pending": VIDEO_JOBS.pending_count(),
    }


@app.post("/api/video/jobs", tags=["video"])
async def create_job(file: UploadFile = File(...),
                     threshold: float | None = Form(None), margin: float | None = Form(None),
                     det_confidence: float = Form(0.25), frame_stride: int = Form(1),
                     reid_interval: int = Form(3), min_votes: int = Form(2),
                     imgsz: int = Form(640), max_frames: int = Form(0)):
    cfg, engine = S.cfg, S.engine
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(415, f"Unsupported video type '{ext or '?'}'. Allowed: {', '.join(sorted(VIDEO_EXTENSIONS))}")
    try:
        params = video_job_params(default_threshold=engine.matcher.threshold, default_margin=engine.matcher.margin,
                                  threshold=threshold, margin=margin, det_confidence=det_confidence,
                                  frame_stride=frame_stride, reid_interval=reid_interval, min_votes=min_votes,
                                  imgsz=imgsz, max_frames=max_frames)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    try:
        VIDEO_JOBS.detector()
    except DetectorUnavailableError as e:
        raise HTTPException(503, str(e)) from e

    job_id = uuid.uuid4().hex
    jdir = VIDEO_JOBS.root / job_id
    jdir.mkdir(parents=True, exist_ok=True)
    inp = jdir / f"input{ext}"
    max_bytes = int(cfg.get("webapp", {}).get("max_video_upload_mb", 1024)) * 1024 * 1024
    size = 0
    try:
        with open(inp, "wb") as f:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Video larger than {max_bytes // (1024 * 1024)} MB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty upload")
    except HTTPException:
        shutil.rmtree(jdir, ignore_errors=True)
        raise
    except OSError as e:
        shutil.rmtree(jdir, ignore_errors=True)
        log.error("Upload could not be stored: %s", e)
        raise HTTPException(507, "The server could not store the upload (disk full or not writable).") from e
    finally:
        await file.close()
    job = Job(job_id, Path(file.filename).name, jdir, inp, params)
    job.progress = {"upload_mb": round(size / 1e6, 1)}
    VIDEO_JOBS.submit(job)
    return job.to_dict()


@app.get("/api/video/jobs/{job_id}", tags=["video"])
def job_status(job_id: str):
    return _job_or_404(job_id).to_dict()


@app.post("/api/video/jobs/{job_id}/cancel", tags=["video"])
def cancel_job(job_id: str):
    job = _job_or_404(job_id)
    job.cancel.set()
    return {"id": job_id, "state": job.state, "cancel_requested": True}


@app.get("/api/video/jobs/{job_id}/files/{name}", tags=["video"])
def job_file(job_id: str, name: str, download: bool = False):
    job = _job_or_404(job_id)
    if name not in job.files():
        raise HTTPException(404, "unknown file")
    path = job.dir / name
    stem = Path(job.filename).stem
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
                        filename=f"{stem}_{name.removeprefix(OUTPUT_STEM + '_')}" if download else None,
                        content_disposition_type="attachment" if download else "inline")


def run(config_path: str | None = None, host: str | None = None, port: int | None = None) -> int:
    import uvicorn

    setup_logging()
    try:
        init_state(config_path)
    except PS1Error as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    w = S.cfg.get("webapp", {})
    host, port = host or w.get("host", "127.0.0.1"), int(port or w.get("port", 8000))
    print(f"Open http://{host}:{port}  (API docs: /docs, Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="info",
                loop="asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto")
    return 0


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PS1 open-set person ReID backend (FastAPI).")
    ap.add_argument("--config", default=None, help="config YAML (default: $PS1_CONFIG or configs/default.yaml)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    return run(args.config, args.host, args.port)


if __name__ == "__main__":
    sys.exit(cli())
