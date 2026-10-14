from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from source.errors import CheckpointError
from source.utils.device_utils import select_device

log = logging.getLogger(__name__)


def _extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise CheckpointError("Checkpoint does not contain a recognisable model state_dict")


def load_checkpoint_info(path: Path) -> dict:
    ckpt = _load_raw(path)
    sd = _extract_state_dict(ckpt)
    return {
        "top_level_keys": list(ckpt.keys()) if isinstance(ckpt, dict) else None,
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "val_accuracy": ckpt.get("val_accuracy") if isinstance(ckpt, dict) else None,
        "num_tensors": len(sd),
        "fc_weight_shape": tuple(sd["fc.weight"].shape) if "fc.weight" in sd else None,
    }


def _load_raw(path: Path):
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint {path}: {e}") from e


def load_embedding_model(
    checkpoint: Path,
    num_train_classes: int = 751,
    embedding_dim: int = 512,
    device: str | torch.device = "auto",
) -> tuple[nn.Module, torch.device]:
    sd = _extract_state_dict(_load_raw(checkpoint))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    fc_w = sd.get("fc.weight")
    if fc_w is None:
        raise CheckpointError("Checkpoint has no fc.weight; it is not the expected ResNet18 classifier")
    if tuple(fc_w.shape) != (num_train_classes, embedding_dim):
        raise CheckpointError(
            f"fc.weight shape {tuple(fc_w.shape)} != expected ({num_train_classes}, {embedding_dim}). "
            "Update model.num_train_classes / embedding_dim in the config."
        )

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(embedding_dim, num_train_classes)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise CheckpointError(f"Checkpoint is not a torchvision ResNet18: {e}") from e

    model.fc = nn.Identity()
    dev = device if isinstance(device, torch.device) else select_device(device)
    model.to(dev).eval()
    log.info("Loaded ResNet18 embedding model from %s on %s", checkpoint, dev)
    return model, dev
