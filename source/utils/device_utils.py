from __future__ import annotations

import logging

log = logging.getLogger("ps1.device")


def select_device(name: str = "auto") -> "torch.device":
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    requested = str(requested or "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{requested}' requested but torch has no CUDA support "
                           f"(torch {torch.__version__}).")
    return requested
