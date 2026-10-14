from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from source.models.yolo_detector.detector import Detection, PersonDetector
from source.logic.reid.open_set import MatchResult, OpenSetMatcher
from source.logic.preprocessing import load_image
from source.logic.reid.engine import ReIDEngine


@dataclass
class PersonResult:
    box: tuple[int, int, int, int] | None
    det_confidence: float | None
    match: MatchResult
    crop: Image.Image


def identify_image(engine: ReIDEngine, image, detect: bool = False, detector: PersonDetector | None = None,
                   top_k: int | None = None, matcher: OpenSetMatcher | None = None) -> tuple[Image.Image, list[PersonResult]]:
    img = load_image(image)
    if detect:
        if detector is None:
            raise ValueError("detect=True requires a PersonDetector")
        dets: list[Detection] = detector.detect(img)
        if not dets:
            return img, []
        crops = [d.crop(img) for d in dets]
        matches = engine.identify_many(crops, top_k=top_k, matcher=matcher)
        return img, [PersonResult(d.box, d.confidence, m, c) for d, m, c in zip(dets, matches, crops)]
    match = engine.identify(img, top_k=top_k, matcher=matcher)
    return img, [PersonResult(None, None, match, img)]
