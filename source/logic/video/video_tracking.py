from __future__ import annotations

import csv
import json
import logging
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from source.logic.video.annotate import VideoSink, draw_hud, draw_tracks, output_size
from source.models.yolo_detector.detector import PersonDetector
from source.errors import ImageError
from source.logic.reid.open_set import OpenSetMatcher
from source.logic.reid.engine import ReIDEngine
from source.logic.video.track_identity import IdentityPolicy, Observation, TrackIdentityManager
from source.logic.video.tracker import BYTETRACK_DEFAULTS, PersonTracker

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm"}


@dataclass
class TrackingSettings:
    frame_stride: int = 1
    max_frames: int = 0
    det_confidence: float = 0.25
    det_low_confidence: float = 0.1
    imgsz: int = 640
    output_max_side: int = 1280
    tracker: dict = field(default_factory=dict)
    identity: IdentityPolicy = field(default_factory=IdentityPolicy)

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "TrackingSettings":
        v = dict(cfg.get("video_tracking", {}) or {})
        ident = IdentityPolicy.from_config({**v.pop("identity", {}), **overrides.pop("identity", {})})
        v.update({k: val for k, val in overrides.items() if val is not None})
        unknown = set(v) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown video_tracking settings: {sorted(unknown)}")
        s = cls(identity=ident, **v)
        if s.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if not 0.0 < s.det_low_confidence <= s.det_confidence < 1.0:
            raise ValueError("need 0 < det_low_confidence <= det_confidence < 1")
        return s

    def bytetrack_args(self) -> dict:
        args = {**BYTETRACK_DEFAULTS, "track_high_thresh": self.det_confidence,
                "new_track_thresh": self.det_confidence, "track_low_thresh": self.det_low_confidence}
        args.update(self.tracker or {})
        return args


@dataclass
class TrackRow:
    frame_index: int
    time_s: float
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    det_confidence: float
    status: str
    identity: str
    score: float | None
    votes: int
    window_observations: int


def open_video(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ImageError(f"OpenCV could not open video '{path.name}' (unsupported container/codec or corrupt file)")
    return cap


def _stats(values) -> dict | None:
    values = list(values)
    if not values:
        return None
    return {"min": int(min(values)), "median": float(statistics.median(values)), "max": int(max(values))}


def track_video(video_path: str | Path, engine: ReIDEngine, detector: PersonDetector, out_dir: str | Path,
                settings: TrackingSettings | None = None, matcher: OpenSetMatcher | None = None,
                progress: Callable[[dict], None] | None = None, cancel: threading.Event | None = None,
                write_video: bool = True, output_stem: str | None = None) -> dict:
    s = settings or TrackingSettings()
    matcher = matcher or engine.matcher
    video_path, out_dir = Path(video_path), Path(out_dir)
    stem = output_stem or video_path.stem
    cap = open_video(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = float(src_fps) if src_fps and np.isfinite(src_fps) and 0 < src_fps < 1000 else 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    est_total = (n_reported + s.frame_stride - 1) // s.frame_stride if n_reported > 0 else 0
    if s.max_frames:
        est_total = min(est_total, s.max_frames) if est_total else s.max_frames
    log.info("Opened %s: %dx%d @ %.2f fps, %s frames reported", video_path.name, w, h, src_fps, n_reported or "?")

    tracker = PersonTracker(**s.bytetrack_args())
    ids = TrackIdentityManager(engine, matcher, s.identity)
    sink = None
    (ow, oh), scale = output_size(w, h, s.output_max_side)
    video_out = out_dir / f"{stem}_tracked.mp4"
    tracks_csv = out_dir / f"{stem}_tracks.csv"
    obs_csv = out_dir / f"{stem}_reid_observations.csv"

    t_det = t_trk = t_reid = t_write = 0.0
    idx = processed = n_dets = n_tracked_boxes = n_accepted = 0
    prev_ids: set[int] = set()
    continuity_hits = continuity_total = 0
    max_active = 0
    cancelled = False
    t0 = time.perf_counter()
    last_progress = 0.0
    try:
        f_tracks = open(tracks_csv, "w", newline="", encoding="utf-8")
        f_obs = open(obs_csv, "w", newline="", encoding="utf-8")
        with f_tracks, f_obs:
            w_tracks = csv.DictWriter(f_tracks, fieldnames=list(TrackRow.__dataclass_fields__))
            w_obs = csv.DictWriter(f_obs, fieldnames=list(Observation.__dataclass_fields__))
            w_tracks.writeheader()
            w_obs.writeheader()
            if write_video:
                sink = VideoSink(video_out, src_fps / s.frame_stride, (ow, oh))
            while True:
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % s.frame_stride:
                    idx += 1
                    continue
                if frame is None or frame.ndim != 3 or frame.shape[0] != h or frame.shape[1] != w:
                    raise ImageError(f"Frame {idx} has unexpected shape {None if frame is None else frame.shape}")

                t = time.perf_counter()
                xyxy, conf = detector.detect_all(frame, s.det_low_confidence, s.imgsz)
                t_det += time.perf_counter() - t
                n_dets += int((conf >= s.det_confidence).sum())

                t = time.perf_counter()
                tracks = tracker.update(xyxy, conf)
                t_trk += time.perf_counter() - t

                t = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                labels, new_obs = ids.step(idx, processed, rgb, tracks)
                t_reid += time.perf_counter() - t

                cur_ids = {tr.track_id for tr in tracks}
                continuity_total += len(prev_ids)
                continuity_hits += len(prev_ids & cur_ids)
                prev_ids = cur_ids
                max_active = max(max_active, len(cur_ids))
                n_tracked_boxes += len(tracks)

                time_s = idx / src_fps
                for tr in tracks:
                    lab = labels[tr.track_id]
                    w_tracks.writerow(asdict(TrackRow(idx, round(time_s, 3), tr.track_id, *tr.box, round(tr.score, 4),
                                                      lab.status, lab.identity,
                                                      None if lab.score is None else round(lab.score, 4),
                                                      lab.votes, lab.n_obs)))
                for o in new_obs:
                    w_obs.writerow(asdict(o))
                    n_accepted += o.decision == "KNOWN"

                if sink is not None:
                    t = time.perf_counter()
                    out = frame if scale == 1.0 else cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
                    draw_tracks(out, tracks, labels, scale)
                    draw_hud(out, [f"frame {idx}  t={time_s:6.2f}s  tracks {len(tracks)}",
                                   f"thr {matcher.threshold:.3f}  margin {matcher.margin:.3f}  (cosine, not probability)"])
                    sink.write(out)
                    t_write += time.perf_counter() - t

                processed += 1
                idx += 1
                now = time.perf_counter()
                if progress is not None and (now - last_progress > 0.5):
                    last_progress = now
                    el = now - t0
                    progress({"frames_processed": processed, "frames_total_est": est_total,
                              "fraction": min(processed / est_total, 1.0) if est_total else None,
                              "elapsed_s": round(el, 1), "processing_fps": round(processed / el, 2) if el else None,
                              "active_tracks": len(tracks), "tracks_so_far": len(ids.states)})
                if s.max_frames and processed >= s.max_frames:
                    break
    finally:
        cap.release()
        if sink is not None:
            sink.close()

    elapsed = time.perf_counter() - t0
    summaries = ids.track_summaries()
    known_tracks = [t for t in summaries if t["final_status"] == "KNOWN"]
    unknown_tracks = [t for t in summaries if t["final_status"] == "UNKNOWN"]
    pending_tracks = [t for t in summaries if t["final_status"] == "PENDING"]
    obs_total = sum(t["reid_observations"] for t in summaries)
    crop_h = [c[0] for c in ids.crop_sizes]
    crop_w = [c[1] for c in ids.crop_sizes]
    g = engine.gallery

    summary = {
        "video": video_path.name,
        "cancelled": cancelled,
        "source": {"resolution": [w, h], "fps": round(src_fps, 3), "frames_reported": n_reported},
        "frames_read": idx, "frames_processed": processed, "frame_stride": s.frame_stride,
        "counts": {
            "person_detections": n_dets,
            "tracked_boxes": n_tracked_boxes,
            "unique_track_ids": len(summaries),
            "tracks_known": len(known_tracks),
            "tracks_unknown": len(unknown_tracks),
            "tracks_insufficient_evidence": len(pending_tracks),
            "matched_known_identities": sorted({t["final_identity"] for t in known_tracks}),
            "reid_observations": obs_total,
            "reid_observations_accepted": n_accepted,
            "max_simultaneous_tracks": max_active,
        },
        "timing": {
            "total_s": round(elapsed, 2),
            "processing_fps": round(processed / elapsed, 2) if elapsed else None,
            "realtime_factor": round((processed / elapsed) / (src_fps / s.frame_stride), 3) if elapsed else None,
            "detector_s": round(t_det, 2), "tracker_s": round(t_trk, 2), "reid_s": round(t_reid, 2),
            "video_write_s": round(t_write, 2),
            "device": str(engine.device),
            "reid_backend": {k: v for k, v in engine.reid.info().items()},
        },
        "diagnostics": {
            "checkpoint": Path(engine.cfg["model"]["checkpoint"]).name,
            "embedding_dim": engine.embedding_dim, "gallery_dim": g.dim,
            "embedding_dim_matches_gallery": engine.embedding_dim == g.dim,
            "gallery_images": len(g), "gallery_identities": int(len(g.identities)),
            "gallery_source": Path(g.source).name,
            "aggregation": engine.cfg["gallery"].get("aggregation"),
            "threshold": matcher.threshold, "margin": matcher.margin,
            "embedded_crops": len(ids.crop_sizes), "embed_batches": ids.embed_batches,
            "crop_height_px": _stats(crop_h), "crop_width_px": _stats(crop_w),
            "crops_skipped": dict(ids.skipped),
            "adjacent_frame_track_continuity": round(continuity_hits / continuity_total, 4) if continuity_total else None,
            "track_length_processed_frames": _stats(t["frames_visible"] for t in summaries),
            "label_changes_total": sum(t["label_changes"] for t in summaries),
            "identity_resets_after_occlusion": ids.identity_resets,
            "bytetrack": s.bytetrack_args(), "identity_policy": asdict(s.identity),
            "imgsz": s.imgsz,
        },
        "outputs": {"video": video_out.name if sink is not None else None,
                    "video_codec": sink.codec if sink is not None else None,
                    "browser_playable": sink.browser_playable if sink is not None else None,
                    "output_resolution": [ow, oh],
                    "tracks_csv": tracks_csv.name, "observations_csv": obs_csv.name},
        "notes": [
            "Track and prediction counts are NOT accuracy metrics; use scripts/evaluate_video_tracking.py with ground truth.",
            "Scores are cosine similarities, not probabilities. The threshold was calibrated on single Market-1501 "
            "images for ~5% unknown false-accept rate; it is not calibrated for this camera or for track-level voting.",
            "A track ID is temporary: the same person gets a new ID after leaving the view or a long occlusion.",
        ],
        "tracks": summaries,
    }
    (out_dir / f"{stem}_tracking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Processed %d frames in %.1fs (%.2f fps): %d tracks, %d known, %d unknown, %d pending",
             processed, elapsed, processed / elapsed if elapsed else 0, len(summaries), len(known_tracks),
             len(unknown_tracks), len(pending_tracks))
    return summary
