from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from source.config import ReIDSettings
from source.errors import BackendError, NoBackendAvailable
from source.models.base import ReIDBackend, l2_normalize
from source.models.onnx_runtime.backend import OnnxBackend
from source.models.reid_resnet18.backend import PyTorchBackend
from source.models.tensorrt.backend import TensorRTBackend

log = logging.getLogger("ps1.reid.select")

ORDER = ("tensorrt", "onnx-cuda", "onnx-dml", "pytorch-cuda", "onnx-cpu", "pytorch-cpu")
CPU_NAMES = {"onnx-cpu", "pytorch-cpu"}


def backend_factories() -> Dict[str, Callable[[ReIDSettings], ReIDBackend]]:
    return {
        "tensorrt": lambda s: TensorRTBackend(s),
        "onnx-cuda": lambda s: OnnxBackend(s, provider="cuda"),
        "onnx-dml": lambda s: OnnxBackend(s, provider="dml"),
        "pytorch-cuda": lambda s: PyTorchBackend(s, device="cuda"),
        "onnx-cpu": lambda s: OnnxBackend(s, provider="cpu"),
        "pytorch-cpu": lambda s: PyTorchBackend(s, device="cpu"),
    }


LEGACY = {"torch": ("pytorch-cuda", "pytorch-cpu"),
          "pytorch": ("pytorch-cuda", "pytorch-cpu"),
          "onnx": ("onnx-cuda", "onnx-dml", "onnx-cpu", "pytorch-cuda", "pytorch-cpu")}


@dataclass
class Candidate:
    name: str
    available: Callable[[], Tuple[bool, str]]
    factory: Callable[[ReIDSettings], ReIDBackend]


def _ort_providers() -> Tuple[Optional[List[str]], str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime is not installed"
    return ort.get_available_providers(), ""


def _torch_cuda() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} has no usable CUDA device"
    return True, ""


def _torch() -> Tuple[bool, str]:
    try:
        import torch
        return True, ""
    except ImportError:
        return False, "torch is not installed"


def _availability(name: str, s: ReIDSettings) -> Callable[[], Tuple[bool, str]]:
    def onnx_file() -> Tuple[bool, str]:
        return (True, "") if s.onnx_path.is_file() else (
            False, f"ONNX model {s.onnx_path.name} not found (run scripts/export_onnx.py)")

    def ckpt() -> Tuple[bool, str]:
        return (True, "") if s.checkpoint.is_file() else (False, f"checkpoint {s.checkpoint.name} not found")

    def ort_provider(provider: str):
        def check():
            providers, why = _ort_providers()
            if providers is None:
                return False, why
            if provider not in providers:
                return False, f"onnxruntime has no {provider} (available: {', '.join(providers)})"
            return onnx_file()
        return check

    def tensorrt():
        from source.models.tensorrt.gpu_runtime import probe_cuda, tensorrt_version
        if tensorrt_version() is None:
            return False, "TensorRT Python package is not installed"
        gpu = probe_cuda()
        if not gpu["available"]:
            return False, f"no CUDA GPU for TensorRT ({gpu['reason']})"
        return onnx_file()

    def both(*checks):
        def check():
            for c in checks:
                ok, why = c()
                if not ok:
                    return ok, why
            return True, ""
        return check

    return {
        "tensorrt": tensorrt,
        "onnx-cuda": ort_provider("CUDAExecutionProvider"),
        "onnx-dml": ort_provider("DmlExecutionProvider"),
        "pytorch-cuda": both(_torch_cuda, ckpt),
        "onnx-cpu": ort_provider("CPUExecutionProvider"),
        "pytorch-cpu": both(_torch, ckpt),
    }[name]


def default_candidates(s: ReIDSettings) -> List[Candidate]:
    factories = backend_factories()
    return [Candidate(n, _availability(n, s), factories[n]) for n in ORDER]


def available_backends(s: ReIDSettings) -> List[str]:
    return [c.name for c in default_candidates(s) if c.available()[0]]


def probe_crops(n: int = 4, seed: int = 1234) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    crops = []
    for i in range(n):
        h, w = int(rng.integers(96, 320)), int(rng.integers(40, 140))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = np.stack([128 + 100 * np.sin(yy / (7 + 3 * i) + c) * np.cos(xx / (5 + i))
                         for c in range(3)], axis=-1)
        crops.append(np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8))
    return crops


def probe_batch(s: ReIDSettings, n: int = 4) -> np.ndarray:
    import torch

    from source.logic.preprocessing import build_transform, load_image

    tr = build_transform(s.preprocessing_dict)
    return torch.stack([tr(load_image(c)) for c in probe_crops(n)]).numpy()


_REFERENCE_CACHE: dict = {}


def _reference_key(s: ReIDSettings):
    try:
        st = s.checkpoint.stat()
    except OSError:
        return None
    return (str(s.checkpoint.resolve()), st.st_size, st.st_mtime_ns, str(s.onnx_path), s.input_size,
            s.preprocessing, s.num_train_classes, s.embedding_dim)


class ReferenceEmbeddings:
    def __init__(self, s: ReIDSettings, compute: Optional[Callable[[], Tuple[str, np.ndarray]]] = None,
                 batch: Optional[np.ndarray] = None):
        self.settings = s
        self.batch = probe_batch(s) if batch is None else batch
        self._compute = compute or self._cached_default
        self._value: Optional[Tuple[str, np.ndarray]] = None
        self.error: Optional[str] = None

    def _cached_default(self) -> Tuple[str, np.ndarray]:
        key = _reference_key(self.settings)
        if key is None or key not in _REFERENCE_CACHE:
            value = self._default()
            if key is not None:
                _REFERENCE_CACHE[key] = value
            return value
        return _REFERENCE_CACHE[key]

    def _default(self) -> Tuple[str, np.ndarray]:
        factories = backend_factories()
        errors = []
        for name in ("pytorch-cpu", "onnx-cpu"):
            ok, why = _availability(name, self.settings)()
            if not ok:
                errors.append(f"{name}: {why}")
                continue
            try:
                b = factories[name](self.settings)
                try:
                    return name, b.embed_preprocessed(self.batch)
                finally:
                    b.close()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise BackendError("no CPU reference available (" + "; ".join(errors) + ")")

    def get(self) -> Optional[Tuple[str, np.ndarray]]:
        if self._value is None and self.error is None:
            try:
                self._value = self._compute()
            except Exception as exc:
                self.error = str(exc)
                log.warning("ReID reference embeddings unavailable: %s", exc)
        return self._value


def verify_backend(backend: ReIDBackend, s: ReIDSettings, reference: ReferenceEmbeddings) -> dict:
    batch = reference.batch
    t0 = time.perf_counter()
    first = backend.embed_preprocessed(batch[:1])
    warm_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb = np.asarray(backend.embed_preprocessed(batch))
    probe_s = time.perf_counter() - t0
    d = s.embedding_dim
    if emb.shape != (len(batch), d) or np.asarray(first).shape != (1, d):
        raise BackendError(f"wrong embedding shape {emb.shape} (expected ({len(batch)}, {d}))")
    if not np.all(np.isfinite(emb)):
        raise BackendError("non-finite values in embeddings")
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise BackendError(f"embeddings are not unit-norm (norms {norms.round(4).tolist()})")
    batch_cos = float(np.sum(l2_normalize(first)[0] * emb[0]))
    if batch_cos < s.verify_min_cosine:
        raise BackendError(f"batch-1 and batch-{len(batch)} results disagree (cos {batch_cos:.6f})")
    report = {"warmup_s": round(warm_s, 4), "probe_batch_s": round(probe_s, 4)}
    ref = reference.get()
    if ref is None:
        report["reference"] = None
        report["reference_note"] = f"not compared: {reference.error}"
        return report
    ref_name, ref_emb = ref
    cos = np.sum(ref_emb * emb, axis=1)
    report.update(reference=ref_name, min_cosine_vs_reference=round(float(cos.min()), 7),
                  max_abs_diff_vs_reference=float(np.abs(ref_emb - emb).max()))
    if float(cos.min()) < s.verify_min_cosine:
        raise BackendError(f"embeddings deviate from the {ref_name} reference (min cosine "
                           f"{cos.min():.6f} < {s.verify_min_cosine}); the model file was probably "
                           f"exported from a different checkpoint than {s.checkpoint.name}")
    return report


@dataclass
class SelectionResult:
    backend: "ResilientBackend"
    attempts: List[dict] = field(default_factory=list)

    def report(self) -> dict:
        return {"selected": self.backend.info(), "attempts": self.attempts}


def _try_candidate(c: Candidate, s: ReIDSettings, reference: ReferenceEmbeddings
                   ) -> Tuple[Optional[ReIDBackend], dict]:
    ok, why = c.available()
    if not ok:
        return None, {"backend": c.name, "status": "unavailable", "reason": why}
    t0 = time.perf_counter()
    backend = None
    try:
        backend = c.factory(s)
        init_s = time.perf_counter() - t0
        check = verify_backend(backend, s, reference)
    except Exception as exc:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        return None, {"backend": c.name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                      "elapsed_s": round(time.perf_counter() - t0, 3)}
    attempt = {"backend": c.name, "status": "selected", "init_s": round(init_s, 3), **check}
    attempt.update({k: v for k, v in backend.info().items() if k != "backend"})
    if getattr(backend, "build_meta", None):
        attempt["engine_built_now"] = True
        attempt["engine_build_time_s"] = backend.build_meta.get("build_time_s")
    return backend, attempt


def resolve_order(requested: str) -> Tuple[str, ...]:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return ORDER
    if requested in LEGACY:
        return LEGACY[requested]
    if requested in ORDER:
        return (requested,)
    raise BackendError(f"Unknown ReID backend '{requested}'. Use 'auto', one of {list(ORDER)}, "
                       f"or a legacy name {sorted(LEGACY)}.")


def select_backend(s: ReIDSettings, candidates: Optional[Sequence[Candidate]] = None,
                   reference: Optional[ReferenceEmbeddings] = None,
                   requested: Optional[str] = None) -> SelectionResult:
    requested = (requested or s.backend or "auto").lower()
    if candidates is None:
        order = resolve_order(requested)
        candidates = [c for c in default_candidates(s) if c.name in order]
        candidates.sort(key=lambda c: order.index(c.name))
    candidates = list(candidates)
    reference = reference or ReferenceEmbeddings(s)
    attempts = []
    for i, c in enumerate(candidates):
        backend, attempt = _try_candidate(c, s, reference)
        attempts.append(attempt)
        log.log(logging.INFO if backend else logging.WARNING, "ReID backend %s: %s%s", c.name,
                attempt["status"], f" ({attempt['reason']})" if "reason" in attempt else "")
        if backend is not None:
            resilient = ResilientBackend(backend, candidates[i + 1:], s, reference, attempts)
            info = resilient.info()
            skipped = attempts[:-1]
            if c.name in CPU_NAMES and any(a["backend"] not in CPU_NAMES for a in skipped):
                log.warning("ReID is running on CPU (%s); GPU backends were not usable: %s", c.name,
                            "; ".join(f"{a['backend']}: {a['reason']}" for a in skipped
                                      if a["backend"] not in CPU_NAMES))
            if requested == "onnx" and not c.name.startswith("onnx"):
                log.warning("ONNX backend unavailable; using %s", c.name)
            log.info("ReID backend selected: %s on %s (precision %s, model %s, init %.2fs, "
                     "min cos vs %s %s)", info["backend"], info["device"], info["precision"],
                     info["model"], attempt["init_s"], attempt.get("reference"),
                     attempt.get("min_cosine_vs_reference"))
            return SelectionResult(resilient, attempts)
    raise NoBackendAvailable("No compatible ReID backend could be loaded: " + "; ".join(
        f"{a['backend']}: {a['reason']}" for a in attempts))


class ResilientBackend:
    def __init__(self, primary: ReIDBackend, fallbacks: Sequence[Candidate], s: ReIDSettings,
                 reference: ReferenceEmbeddings, attempts: List[dict]):
        self.active = primary
        self.settings = s
        self._fallbacks = list(fallbacks)
        self._reference = reference
        self.attempts = attempts
        self.fallback_events: List[dict] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.active.name

    @property
    def device(self) -> str:
        return self.active.device

    def _switch(self, failed: ReIDBackend, exc: Exception):
        with self._lock:
            if self.active is not failed:
                return
            reason = f"{type(exc).__name__}: {exc}"
            log.error("ReID backend %s failed during inference (%s); trying fallback", failed.name, reason)
            while self._fallbacks:
                c = self._fallbacks.pop(0)
                backend, attempt = _try_candidate(c, self.settings, self._reference)
                attempt["trigger"] = f"runtime fallback after {failed.name} failure"
                self.attempts.append(attempt)
                if backend is not None:
                    self.fallback_events.append({"from": failed.name, "to": backend.name,
                                                 "reason": reason, "time": time.time()})
                    try:
                        failed.close()
                    except Exception:
                        pass
                    self.active = backend
                    log.warning("ReID backend switched %s -> %s", failed.name, backend.name)
                    return
            raise NoBackendAvailable(f"ReID backend {failed.name} failed ({reason}) and no "
                                     "fallback backend could be loaded.") from exc

    def embed_preprocessed(self, batch):
        while True:
            backend = self.active
            try:
                return backend.embed_preprocessed(batch)
            except NoBackendAvailable:
                raise
            except Exception as exc:
                self._switch(backend, exc)

    def close(self):
        self.active.close()

    def info(self) -> dict:
        info = dict(self.active.info())
        if self.fallback_events:
            info["fallback_events"] = list(self.fallback_events)
        return info
