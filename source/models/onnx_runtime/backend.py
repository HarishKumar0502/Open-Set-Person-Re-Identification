from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from source.config import ReIDSettings
from source.errors import BackendError
from source.models.base import ReIDBackend


ORT_PROVIDERS = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider",
                 "cpu": "CPUExecutionProvider"}


class OnnxBackend(ReIDBackend):
    name = "onnx"

    def __init__(self, settings: ReIDSettings, path: Optional[Path] = None, provider: str = "auto"):
        super().__init__(settings)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendError("onnxruntime is not installed.") from exc
        path = Path(path or settings.onnx_path)
        if not path.is_file():
            raise BackendError(f"ONNX model not found ({path.name}); run scripts/export_onnx.py.")
        available = ort.get_available_providers()
        if provider == "auto":
            provider = "cuda" if ORT_PROVIDERS["cuda"] in available else "cpu"
        if provider not in ORT_PROVIDERS:
            raise BackendError(f"Unknown ONNX Runtime provider '{provider}'.")
        wanted = ORT_PROVIDERS[provider]
        if wanted not in available:
            raise BackendError(f"ONNX Runtime {ort.__version__} has no {wanted} "
                               f"(available: {', '.join(available)}).")
        providers = [wanted] + (["CPUExecutionProvider"] if provider != "cpu" else [])
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception as exc:
            raise BackendError(f"ONNX Runtime could not create a {wanted} session: {exc}") from exc
        active = self.session.get_providers()
        if active[0] != wanted:
            raise BackendError(f"ONNX Runtime fell back to {active[0]} instead of {wanted}.")
        inp = self.session.get_inputs()[0]
        if tuple(inp.shape[1:]) != (3, *settings.input_size):
            raise BackendError(f"ONNX input shape {inp.shape} does not match the configured "
                               f"ReID input (3, {settings.input_size[0]}, {settings.input_size[1]}).")
        self.device = {"cuda": "cuda", "dml": "directml", "cpu": "cpu"}[provider]
        self.name = f"onnx-{provider}"
        self.model_path = path
        self.input_name = inp.name

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch.astype(np.float32, copy=False)})[0]

    def close(self):
        self.session = None
