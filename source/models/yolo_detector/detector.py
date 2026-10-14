from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from source.errors import DetectorUnavailableError

log = logging.getLogger(__name__)
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop(self.box)


class PersonDetector:
    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.4, min_box_height: int = 48,
                 device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorUnavailableError(
                "Person detection needs the 'ultralytics' package: python -m pip install ultralytics") from e
        try:
            weights = str(weights)
            if not Path(weights).is_file():
                from ultralytics.utils.downloads import attempt_download_asset
                Path(weights).parent.mkdir(parents=True, exist_ok=True)
                weights = str(attempt_download_asset(weights))
            self.model = YOLO(weights)
        except Exception as e:
            raise DetectorUnavailableError(
                f"Could not load YOLO weights '{weights}' (offline? place the .pt file locally): {e}") from e
        self.confidence = float(confidence)
        self.min_box_height = int(min_box_height)
        self.device = device

    def detect(self, image: Image.Image | np.ndarray) -> list[Detection]:
        arr = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else image
        res = self.model.predict(arr[:, :, ::-1], conf=self.confidence, classes=[PERSON_CLASS_ID],
                                 device=self.device, verbose=False)[0]
        h, w = arr.shape[:2]
        out = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
            if (y2 - y1) >= self.min_box_height and (x2 - x1) >= 8:
                out.append(Detection((x1, y1, x2, y2), float(c)))
        return sorted(out, key=lambda d: -d.confidence)

    def detect_all(self, frame_bgr: np.ndarray, confidence: float, imgsz: int = 640) -> tuple[np.ndarray, np.ndarray]:
        res = self.model.predict(frame_bgr, conf=float(confidence), classes=[PERSON_CLASS_ID], imgsz=int(imgsz),
                                 device=self.device, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1)
        h, w = frame_bgr.shape[:2]
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
        return xyxy, conf


def detector_from_config(cfg: dict, device: str = "cpu") -> PersonDetector:
    d = cfg.get("detection", {})
    return PersonDetector(d.get("yolo_weights", "yolov8n.pt"), d.get("confidence", 0.4),
                          d.get("min_box_height", 48), device)
