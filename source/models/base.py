from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from source.config import ReIDSettings

log = logging.getLogger("ps1.reid.backend")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def family(name: str) -> str:
    return "torch" if name.startswith("pytorch") else name.split("-")[0]


class ReIDBackend(ABC):
    name: str = "base"
    device: str = "cpu"
    precision: str = "fp32"
    model_path: Optional[Path] = None

    def __init__(self, settings: ReIDSettings):
        self.settings = settings

    @abstractmethod
    def _infer(self, batch: np.ndarray) -> np.ndarray:
        pass

    def embed_preprocessed(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float32)
        if len(batch) == 0:
            return np.zeros((0, self.settings.embedding_dim), dtype=np.float32)
        bs = self.settings.batch_size
        feats = [self._infer(np.ascontiguousarray(batch[i:i + bs])) for i in range(0, len(batch), bs)]
        return l2_normalize(np.concatenate(feats, axis=0))

    def close(self):
        pass

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device, "precision": self.precision,
                "model": self.model_path.name if self.model_path else None}
