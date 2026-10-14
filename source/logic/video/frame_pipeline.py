from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from source.models.yolo_detector.detector import PersonDetector
from source.errors import ImageError
from source.logic.reid.engine import ReIDEngine

log = logging.getLogger(__name__)


@dataclass
class FrameDetection:
    frame_index: int
    time_s: float
    x1: int
    y1: int
    x2: int
    y2: int
    det_confidence: float
    decision: str
    identity: str
    top1_candidate: str
    top1_score: float
    margin: float | None


def _draw(frame_bgr: np.ndarray, det: FrameDetection) -> None:
    known = det.decision == "KNOWN"
    color = (40, 180, 40) if known else (40, 40, 220)
    cv2.rectangle(frame_bgr, (det.x1, det.y1), (det.x2, det.y2), color, 2)
    label = f"{'ID ' + det.identity if known else 'UNKNOWN'} {det.top1_score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(det.y1, th + 4)
    cv2.rectangle(frame_bgr, (det.x1, y - th - 4), (det.x1 + tw + 4, y), color, -1)
    cv2.putText(frame_bgr, label, (det.x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def process_video(video_path: str | Path, engine: ReIDEngine, detector: PersonDetector, out_dir: str | Path,
                  frame_stride: int = 5, max_frames: int = 0, write_video: bool = True) -> dict:
    video_path, out_dir = Path(video_path), Path(out_dir)
    if not video_path.is_file():
        raise ImageError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ImageError(f"OpenCV could not open video: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if write_video:
        writer = cv2.VideoWriter(str(out_dir / f"{video_path.stem}_annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), max(fps / frame_stride, 1.0), (w, h))

    rows: list[FrameDetection] = []
    idx = processed = 0
    t_det = t_reid = 0.0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                t = time.perf_counter()
                dets = detector.detect(rgb)
                t_det += time.perf_counter() - t
                if dets:
                    t = time.perf_counter()
                    results = engine.identify_many([d.crop(pil) for d in dets])
                    t_reid += time.perf_counter() - t
                    for d, r in zip(dets, results):
                        top1 = r.ranking.identity_ids[0] if len(r.ranking.identity_ids) else ""
                        fd = FrameDetection(idx, idx / fps, *d.box, d.confidence, r.decision, str(r.identity),
                                            str(top1), r.top1_score, r.margin)
                        rows.append(fd)
                        _draw(frame, fd)
                if writer is not None:
                    writer.write(frame)
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    with open(out_dir / f"{video_path.stem}_detections.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(FrameDetection.__dataclass_fields__))
        wr.writeheader()
        wr.writerows(asdict(r) for r in rows)
    elapsed = time.perf_counter() - t0
    summary = {
        "video": str(video_path), "frames_read": idx, "frames_processed": processed, "frame_stride": frame_stride,
        "resolution": [w, h], "source_fps": fps, "detections": len(rows),
        "known_detections": sum(r.decision == "KNOWN" for r in rows),
        "unknown_detections": sum(r.decision == "UNKNOWN" for r in rows),
        "elapsed_s": round(elapsed, 2),
        "processed_fps": round(processed / elapsed, 2) if elapsed else None,
        "detector_time_s": round(t_det, 2), "reid_time_s": round(t_reid, 2),
        "tracking": "none - per-frame independent identification (no multi-object tracker integrated)",
    }
    (out_dir / f"{video_path.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
