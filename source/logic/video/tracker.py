from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from source.errors import DetectorUnavailableError

BYTETRACK_DEFAULTS = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    score: float
    det_index: int


class _Detections:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), np.float32)
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        self.xywh = np.concatenate([self.xyxy[:, :2] + wh / 2, wh], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Detections":
        return _Detections(self.xyxy[idx], self.conf[idx])


class PersonTracker:
    def __init__(self, **overrides):
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace
        except ImportError as e:
            raise DetectorUnavailableError(
                f"ByteTrack is unavailable ({e}). Install: python -m pip install ultralytics \"lap>=0.5.12\"") from e
        unknown = set(overrides) - set(BYTETRACK_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown ByteTrack settings: {sorted(unknown)}")
        self.args = {**BYTETRACK_DEFAULTS, **overrides}
        self._tracker = BYTETracker(IterableSimpleNamespace(tracker_type="bytetrack", **self.args))

    def update(self, xyxy: np.ndarray, conf: np.ndarray) -> list[Track]:
        dets = _Detections(xyxy, conf)
        out = self._tracker.update(dets)
        tracks = []
        for _x1, _y1, _x2, _y2, tid, score, _cls, idx in np.asarray(out).reshape(-1, 8):
            x1, y1, x2, y2 = dets.xyxy[int(idx)]
            tracks.append(Track(int(tid), (int(x1), int(y1), int(round(x2)), int(round(y2))), float(score), int(idx)))
        return tracks

    def reset(self) -> None:
        self._tracker.reset()
