from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms as T

from source.errors import ImageError

def _PIL_OPEN(fp):
    patches = sys.modules.get("ultralytics.utils.patches")
    return (getattr(patches, "_image_open", None) or Image.open)(fp)


def build_transform(pre_cfg: dict) -> T.Compose:
    interp = getattr(T.InterpolationMode, str(pre_cfg.get("interpolation", "bilinear")).upper())
    return T.Compose([
        T.Resize((int(pre_cfg["height"]), int(pre_cfg["width"])), interpolation=interp),
        T.ToTensor(),
        T.Normalize(pre_cfg["mean"], pre_cfg["std"]),
    ])


def load_image(src) -> Image.Image:
    try:
        if isinstance(src, Image.Image):
            img = src
        elif isinstance(src, (bytes, bytearray)):
            img = _PIL_OPEN(io.BytesIO(src))
        elif isinstance(src, np.ndarray):
            if src.ndim != 3 or src.shape[2] != 3:
                raise ImageError(f"Expected HxWx3 RGB array, got shape {src.shape}")
            img = Image.fromarray(src.astype(np.uint8))
        else:
            p = Path(src)
            if not p.is_file():
                raise ImageError(f"Image not found: {p}")
            img = _PIL_OPEN(p)
        img.load()
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise ImageError(f"Could not decode image: {e}") from e
    if img.width < 8 or img.height < 8:
        raise ImageError(f"Image too small ({img.width}x{img.height})")
    return img.convert("RGB")
