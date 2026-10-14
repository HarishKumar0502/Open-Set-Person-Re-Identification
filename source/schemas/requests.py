from __future__ import annotations

from source.logic.reid.open_set import OpenSetMatcher

IMGSZ_CHOICES = (320, 480, 640, 800, 960, 1280)


def video_job_params(*, default_threshold: float, default_margin: float, threshold: float | None,
                     margin: float | None, det_confidence: float, frame_stride: int, reid_interval: int,
                     min_votes: int, imgsz: int, max_frames: int) -> dict:
    threshold = default_threshold if threshold is None else float(threshold)
    margin = default_margin if margin is None else float(margin)
    OpenSetMatcher(threshold, margin)
    checks = [(0.01 <= det_confidence <= 0.95, "det_confidence must be in [0.01, 0.95]"),
              (1 <= frame_stride <= 30, "frame_stride must be in [1, 30]"),
              (1 <= reid_interval <= 300, "reid_interval must be in [1, 300]"),
              (1 <= min_votes <= 8, "min_votes must be in [1, 8]"),
              (imgsz in IMGSZ_CHOICES, "imgsz must be one of 320/480/640/800/960/1280"),
              (0 <= max_frames <= 1_000_000, "max_frames must be >= 0")]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)
    return {"threshold": threshold, "margin": margin,
            "det_confidence": float(det_confidence), "det_low_confidence": min(0.1, float(det_confidence)),
            "frame_stride": int(frame_stride), "reid_interval": int(reid_interval),
            "min_votes": int(min_votes), "imgsz": int(imgsz), "max_frames": int(max_frames)}
