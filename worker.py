from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from source.errors import PS1Error
from source.models.yolo_detector.detector import PersonDetector, detector_from_config
from source.logic.reid.engine import ReIDEngine
from source.logic.reid.open_set import OpenSetMatcher
from source.logic.video.video_tracking import TrackingSettings, track_video

log = logging.getLogger("ps1.worker")
OUTPUT_STEM = "video"
OUTPUT_SUFFIXES = ("_tracked.mp4", "_tracks.csv", "_reid_observations.csv", "_tracking_summary.json")


@dataclass
class Job:
    id: str
    filename: str
    dir: Path
    input_path: Path
    params: dict
    state: str = "queued"
    created: float = field(default_factory=time.time)
    progress: dict = field(default_factory=dict)
    summary: dict | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def files(self) -> dict[str, str]:
        if not self.summary:
            return {}
        o = self.summary["outputs"]
        names = [o.get("video"), o.get("tracks_csv"), o.get("observations_csv"), f"{OUTPUT_STEM}_tracking_summary.json"]
        return {n: f"/api/video/jobs/{self.id}/files/{n}" for n in names if n and (self.dir / n).is_file()}

    def to_dict(self) -> dict:
        d = {"id": self.id, "filename": self.filename, "state": self.state, "params": self.params,
             "progress": self.progress, "error": self.error, "files": self.files()}
        if self.summary:
            s = dict(self.summary)
            tracks = s.pop("tracks", [])
            d["summary"] = s
            d["tracks"] = tracks[:500]
            d["tracks_truncated"] = len(tracks) > 500
        return d


class VideoJobs:
    def __init__(self, get_cfg: Callable[[], dict], get_engine: Callable[[], ReIDEngine]):
        self.get_cfg, self.get_engine = get_cfg, get_engine
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-job")
        self._detector: PersonDetector | None = None
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return Path(self.get_cfg().get("webapp", {}).get("video_jobs_dir", "outputs/video_jobs"))

    def detector(self) -> PersonDetector:
        with self._lock:
            if self._detector is None:
                self._detector = detector_from_config(self.get_cfg(), str(self.get_engine().device))
            return self._detector

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def pending_count(self) -> int:
        return sum(j.state in ("queued", "running") for j in self.jobs.values())

    def submit(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self.pool.submit(self._run, job)
        return job

    def sweep_stale_inputs(self) -> int:
        removed = 0
        if self.root.is_dir():
            for f in self.root.glob("*/input.*"):
                if f.parent.name not in self.jobs:
                    f.unlink(missing_ok=True)
                    removed += 1
        if removed:
            log.info("Removed %d stale uploaded input(s) from %s", removed, self.root)
        return removed

    def shutdown(self) -> None:
        for job in self.jobs.values():
            if job.state in ("queued", "running"):
                job.cancel.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
        for job in self.jobs.values():
            if job.state == "queued":
                job.state = "cancelled"
                job.input_path.unlink(missing_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-job")

    def _discard_outputs(self, job: Job) -> None:
        for suffix in OUTPUT_SUFFIXES:
            (job.dir / f"{OUTPUT_STEM}{suffix}").unlink(missing_ok=True)

    def _run(self, job: Job) -> None:
        if job.cancel.is_set():
            job.state = "cancelled"
            job.input_path.unlink(missing_ok=True)
            return
        job.state = "running"
        t0 = time.time()
        try:
            cfg, engine = self.get_cfg(), self.get_engine()
            p = job.params
            matcher = OpenSetMatcher(p["threshold"], p["margin"])
            settings = TrackingSettings.from_config(
                cfg, frame_stride=p["frame_stride"], max_frames=p["max_frames"], det_confidence=p["det_confidence"],
                det_low_confidence=p["det_low_confidence"], imgsz=p["imgsz"], identity={"reid_interval": p["reid_interval"], "min_votes": p["min_votes"]})
            log.info("job %s: %s params=%s", job.id, job.filename, p)
            job.summary = track_video(job.input_path, engine, self.detector(), job.dir, settings, matcher,
                                      progress=lambda pr: job.progress.update(pr), cancel=job.cancel,
                                      output_stem=OUTPUT_STEM)
            job.summary["video"] = job.filename
            job.state = "cancelled" if job.summary.get("cancelled") else "done"
            log.info("job %s %s in %.1fs: %s", job.id, job.state, time.time() - t0, job.summary["counts"])
        except (PS1Error, ValueError, OSError) as e:
            job.state, job.error = "error", str(e)
            self._discard_outputs(job)
            log.warning("job %s failed: %s", job.id, e)
        except Exception as e:
            job.state, job.error = "error", f"{type(e).__name__}: {e}"
            self._discard_outputs(job)
            log.exception("job %s crashed", job.id)
        finally:
            job.input_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    from source.config import load_config, resolve_path
    from source.logic.video.video_tracking import VIDEO_EXTENSIONS
    from source.schemas.requests import video_job_params
    from source.utils.logging_config import setup_logging

    ap = argparse.ArgumentParser(description="Start the backend server, or run one video job with --video.")
    ap.add_argument("--video", default=None, help="run one job on this video instead of starting the server")
    ap.add_argument("--host", default=None, help="server host (default: webapp.host from the config)")
    ap.add_argument("--port", type=int, default=None, help="server port (default: webapp.port from the config)")
    ap.add_argument("--out", default=None, help="jobs directory (default: webapp.video_jobs_dir from the config)")
    ap.add_argument("--config", default=None, help="config YAML (default: $PS1_CONFIG or configs/default.yaml)")
    ap.add_argument("--threshold", type=float, default=None, help="open-set threshold (default: calibrated value)")
    ap.add_argument("--margin", type=float, default=None, help="top1-top2 margin (default: calibrated value)")
    ap.add_argument("--det-confidence", type=float, default=0.25)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--reid-interval", type=int, default=3)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    args = ap.parse_args(argv)
    if args.video is None:
        from source.main import run
        return run(args.config, args.host, args.port)
    setup_logging()

    video = Path(args.video)
    if not video.is_file():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        return 2
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"ERROR: unsupported video type '{video.suffix}'. Allowed: {', '.join(sorted(VIDEO_EXTENSIONS))}",
              file=sys.stderr)
        return 2
    try:
        cfg = load_config(args.config)
        if args.out:
            cfg.setdefault("webapp", {})["video_jobs_dir"] = resolve_path(args.out)
        engine = ReIDEngine(cfg)
    except PS1Error as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    jobs = VideoJobs(lambda: cfg, lambda: engine)
    try:
        params = video_job_params(default_threshold=engine.matcher.threshold, default_margin=engine.matcher.margin,
                                  threshold=args.threshold, margin=args.margin, det_confidence=args.det_confidence,
                                  frame_stride=args.frame_stride, reid_interval=args.reid_interval,
                                  min_votes=args.min_votes, imgsz=args.imgsz, max_frames=args.max_frames)
        jobs.detector()
        job_id = uuid.uuid4().hex
        jdir = jobs.root / job_id
        jdir.mkdir(parents=True, exist_ok=True)
        inp = jdir / f"input{video.suffix.lower()}"
        shutil.copy2(video, inp)
        job = jobs.submit(Job(job_id, video.name, jdir, inp, params))
        last = None
        try:
            while job.state in ("queued", "running"):
                if job.progress != last:
                    last = dict(job.progress)
                    log.info("job %s: %s %s", job_id, job.state, last)
                time.sleep(1.0)
        except KeyboardInterrupt:
            job.cancel.set()
            log.warning("Cancelling job %s ...", job_id)
    except (ValueError, PS1Error, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        jobs.shutdown()
        engine.close()
    result = {"id": job.id, "state": job.state, "error": job.error, "dir": str(job.dir),
              "outputs": sorted(job.files()),
              "counts": (job.summary or {}).get("counts")}
    print(json.dumps(result, indent=2))
    return 0 if job.state == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
