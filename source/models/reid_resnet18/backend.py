from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from source.config import ReIDSettings
from source.errors import BackendError
from source.models.base import ReIDBackend
from source.utils.device_utils import resolve_device


class PyTorchBackend(ReIDBackend):
    name = "pytorch"

    def __init__(self, settings: ReIDSettings, device: Optional[str] = None):
        super().__init__(settings)
        import torch

        from source.models.reid_resnet18.model import load_embedding_model

        self.torch = torch
        try:
            self.device = resolve_device(device or settings.device)
        except RuntimeError as exc:
            raise BackendError(str(exc)) from exc
        self.name = f"pytorch-{'cuda' if self.device.startswith('cuda') else 'cpu'}"
        self.model_path = Path(settings.checkpoint)
        self.model, _ = load_embedding_model(settings.checkpoint, settings.num_train_classes,
                                             settings.embedding_dim, torch.device(self.device))

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device, non_blocking=True)
            return self.model(x).float().cpu().numpy()

    def close(self):
        self.model = None
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
