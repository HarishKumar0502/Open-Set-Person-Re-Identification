from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from source.config import ReIDSettings, file_sha256
from source.errors import ArtifactError, ConfigError
from source.logic.model_manager import SelectionResult, select_backend
from source.models.base import family
from source.logic.preprocessing import build_transform, load_image
from source.logic.reid.gallery import Gallery
from source.logic.reid.open_set import MatchResult, OpenSetMatcher
from source.utils.device_utils import resolve_device

log = logging.getLogger("ps1.reid.engine")

GALLERY_MIN_COSINE = 0.999


class ReIDEngine:
    def __init__(self, cfg: dict, gallery: Gallery | None = None, backend=None):
        self.cfg = cfg
        self.settings = ReIDSettings.from_config(cfg)
        self.embedding_dim = self.settings.embedding_dim
        try:
            self.device = torch.device(resolve_device(self.settings.device))
        except RuntimeError:
            self.device = torch.device("cpu")
        mode = str(cfg["gallery"].get("consistency_check", "strict")).lower()
        if mode not in ("strict", "warn", "off"):
            raise ConfigError(f"gallery.consistency_check must be strict, warn or off, got '{mode}'")
        self._check_threshold_binding(mode)

        self.selection: Optional[SelectionResult] = None
        if backend is None:
            self.selection = select_backend(self.settings)
            backend = self.selection.backend
        self.reid = backend
        self.transform = build_transform(cfg["preprocessing"])
        self.gallery = gallery if gallery is not None else Gallery.from_npz(cfg["data"]["gallery_embeddings"])
        if self.gallery.dim != self.embedding_dim:
            raise ValueError(f"Gallery dim {self.gallery.dim} != model embedding dim {self.embedding_dim}")
        self.matcher = OpenSetMatcher.from_config(cfg)
        self.consistency = self._check_gallery(mode)

    @property
    def backend(self) -> str:
        return family(self.reid.name)

    @property
    def backend_info(self) -> dict:
        info = dict(self.reid.info())
        if self.selection is not None:
            info["selection"] = self.selection.attempts
        return info

    def close(self) -> None:
        self.reid.close()

    def _checkpoint_sha(self) -> str:
        return file_sha256(self.settings.checkpoint)

    def _check_threshold_binding(self, mode: str) -> None:
        want = self.cfg["open_set"].get("calibrated_for_sha256")
        if mode == "off" or not want or not Path(self.settings.checkpoint).is_file():
            return
        have = self._checkpoint_sha()
        if have != str(want).lower():
            raise ConfigError(
                f"open_set.threshold was calibrated for checkpoint sha256 {str(want)[:12]}..., but "
                f"{Path(self.settings.checkpoint).name} has sha256 {have[:12]}... . A threshold does not transfer "
                "between models: use the matching profile in configs/profiles/ or re-calibrate with "
                "scripts/calibrate_threshold.py.")

    def _gallery_image_paths(self, rows: np.ndarray) -> list[Path]:
        root = self.gallery.meta.get("image_root")
        if root:
            root = Path(root)
        elif "market1501_root" in self.cfg["data"]:
            root = Path(self.cfg["data"]["market1501_root"]) / "bounding_box_test"
        else:
            return []
        return [root / str(self.gallery.filenames[i]) for i in rows]

    def _check_gallery(self, mode: str, n: int = 8) -> dict:
        if mode == "off":
            return {"status": "skipped", "reason": "gallery.consistency_check is off"}
        if self.gallery.source == "memory" or len(self.gallery) == 0:
            return {"status": "skipped", "reason": "in-memory gallery"}
        stored = self.gallery.meta.get("model_sha256")
        if stored:
            ok = stored == self._checkpoint_sha()
            result = {"status": "verified" if ok else "mismatch", "method": "model_sha256"}
            if not ok:
                result["reason"] = (f"gallery was built with checkpoint sha256 {stored[:12]}..., the configured "
                                    f"checkpoint is {self._checkpoint_sha()[:12]}...")
        else:
            rows = np.unique(np.linspace(0, len(self.gallery) - 1, n).astype(int))
            paths = [p for p in self._gallery_image_paths(rows)]
            if not paths or not all(p.is_file() for p in paths):
                log.warning("Gallery %s has no model fingerprint and its images are not available; "
                            "cannot verify that it matches %s", Path(self.gallery.source).name,
                            Path(self.settings.checkpoint).name)
                return {"status": "unverified", "reason": "no model_sha256 in the gallery and images not found"}
            cos = (self.embed(paths) * self.gallery.features[rows]).sum(1)
            ok = float(cos.min()) >= GALLERY_MIN_COSINE
            result = {"status": "verified" if ok else "mismatch", "method": "re-embedded gallery images",
                      "images": len(rows), "min_cosine": round(float(cos.min()), 6)}
            if not ok:
                result["reason"] = (f"re-embedding {len(rows)} gallery images gives min cosine {cos.min():.4f} "
                                    f"< {GALLERY_MIN_COSINE} against the stored embeddings")
        if result["status"] == "mismatch":
            msg = (f"Gallery {Path(self.gallery.source).name} does not belong to checkpoint "
                   f"{Path(self.settings.checkpoint).name}: {result['reason']}. Rebuild it with "
                   "scripts/build_gallery.py / scripts/extract_embeddings.py, or use the matching profile.")
            if mode == "strict":
                raise ArtifactError(msg)
            log.warning(msg)
        return result

    @torch.inference_mode()
    def preprocess(self, images: Sequence) -> np.ndarray:
        return torch.stack([self.transform(load_image(x)) for x in images]).numpy()

    def embed(self, images: Sequence, batch_size: int = 64) -> np.ndarray:
        images = list(images)
        out = [self.reid.embed_preprocessed(self.preprocess(images[i:i + batch_size]))
               for i in range(0, len(images), batch_size)]
        if not out:
            return np.zeros((0, self.embedding_dim), np.float32)
        return np.concatenate(out).astype(np.float32)

    def embed_one(self, image) -> np.ndarray:
        return self.embed([image])[0]

    def identify(self, image, top_k: int | None = None, exclude_mask: np.ndarray | None = None,
                 matcher: OpenSetMatcher | None = None) -> MatchResult:
        return self.identify_embedding(self.embed_one(image), top_k, exclude_mask, matcher)

    def identify_embedding(self, emb: np.ndarray, top_k: int | None = None,
                           exclude_mask: np.ndarray | None = None,
                           matcher: OpenSetMatcher | None = None) -> MatchResult:
        top_k = top_k or int(self.cfg["gallery"].get("top_k", 10))
        ranking = self.gallery.search(emb, top_k=top_k, exclude_mask=exclude_mask,
                                      aggregation=self.cfg["gallery"].get("aggregation", "max"))
        return (matcher or self.matcher).decide(ranking)

    def identify_many(self, images: Iterable, top_k: int | None = None,
                      matcher: OpenSetMatcher | None = None) -> list[MatchResult]:
        embs = self.embed(list(images))
        return [self.identify_embedding(e, top_k, matcher=matcher) for e in embs]
