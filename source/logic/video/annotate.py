from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from source.logic.video.track_identity import PENDING, TrackLabel
from source.logic.video.tracker import Track

log = logging.getLogger(__name__)

KNOWN_COLOR = (60, 170, 40)
UNKNOWN_COLOR = (40, 40, 210)
PENDING_COLOR = (150, 150, 150)


class VideoSink:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], prefer_h264: bool = True):
        self.path = Path(path)
        self.size = size
        self.frames = 0
        self._gen = self._cv = None
        if prefer_h264:
            try:
                import imageio_ffmpeg
                self._gen = imageio_ffmpeg.write_frames(
                    str(self.path), size, fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                    macro_block_size=1, ffmpeg_log_level="error",
                    output_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-preset", "veryfast", "-crf", "23",
                                   "-movflags", "+faststart"])
                self._gen.send(None)
                self.codec, self.browser_playable = "h264 (libx264 via ffmpeg)", True
            except Exception as e:
                log.warning("ffmpeg H.264 writer unavailable (%s); falling back to OpenCV mp4v", e)
                self._gen = None
        if self._gen is None:
            self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self._cv.isOpened():
                raise OSError(f"Could not open a video writer for {self.path}")
            self.codec, self.browser_playable = "mpeg4 (OpenCV mp4v; may not play inline in browsers)", False

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1::-1] != self.size:
            raise ValueError(f"frame size {frame_bgr.shape[1::-1]} != writer size {self.size}")
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(frame_bgr))
        else:
            self._cv.write(frame_bgr)
        self.frames += 1

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None


def output_size(width: int, height: int, max_side: int) -> tuple[tuple[int, int], float]:
    s = 1.0 if not max_side or max(width, height) <= max_side else max_side / max(width, height)
    return (max(2, int(round(width * s))), max(2, int(round(height * s)))), s


def label_text(track_id: int, label: TrackLabel) -> str:
    return f"Track {track_id} | {label.text}"


def draw_tracks(frame: np.ndarray, tracks: list[Track], labels: dict[int, TrackLabel], scale: float = 1.0) -> None:
    h = frame.shape[0]
    fs = max(0.4, min(0.8, h / 900))
    th = 1 if h < 900 else 2
    order = {PENDING: 0, "UNKNOWN": 1, "KNOWN": 2}
    for t in sorted(tracks, key=lambda t: order.get(getattr(labels.get(t.track_id), "status", PENDING), 0)):
        lab = labels.get(t.track_id)
        if lab is None:
            continue
        x1, y1, x2, y2 = (int(round(v * scale)) for v in t.box)
        color = KNOWN_COLOR if lab.status == "KNOWN" else PENDING_COLOR if lab.status == PENDING else UNKNOWN_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if lab.status == "KNOWN" else 2)
        text = label_text(t.track_id, lab)
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y1 - 4 if y1 - tht - base - 4 >= 0 else y1 + tht + base + 4
        tx = min(max(0, x1), max(0, frame.shape[1] - tw - 4))
        cv2.rectangle(frame, (tx, ty - tht - base), (tx + tw + 4, ty + base // 2), color, -1)
        cv2.putText(frame, text, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    fs = max(0.4, min(0.6, frame.shape[0] / 1100))
    sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0] for s in lines]
    if not sizes:
        return
    lh = max(s[1] for s in sizes) + 8
    w = max(s[0] for s in sizes) + 12
    roi = frame[0:lh * len(lines) + 6, 0:w]
    roi[:] = (roi * 0.45).astype(np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (6, lh * (i + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
