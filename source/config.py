from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

import yaml

from source.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

_PATH_KEYS = {
    ("model", "checkpoint"),
    ("model", "onnx_path"),
    ("model", "tensorrt_dir"),
    ("data", "market1501_root"),
    ("data", "query_embeddings"),
    ("data", "gallery_embeddings"),
    ("detection", "yolo_weights"),
    ("webapp", "video_jobs_dir"),
}

ENV_OVERRIDES = {
    ("model", "backend"): "PS1_REID_BACKEND",
    ("model", "device"): "PS1_DEVICE",
    ("model", "tensorrt_precision"): "PS1_TENSORRT_PRECISION",
    ("model", "tensorrt_auto_build"): "PS1_TENSORRT_AUTO_BUILD",
    ("webapp", "host"): "PS1_HOST",
    ("webapp", "port"): "PS1_PORT",
}


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def default_config_path() -> Path:
    return resolve_path(os.environ["PS1_CONFIG"]) if os.environ.get("PS1_CONFIG") else DEFAULT_CONFIG


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read_yaml(path: Path, seen: tuple = ()) -> dict:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    if path in seen:
        raise ConfigError(f"Circular 'extends' in {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(_read_yaml((path.parent / parent).resolve(), seen + (path,)), cfg)
    return cfg


def _coerce(raw: str, current):
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path) if path else default_config_path()
    cfg = _read_yaml(path)

    for (section, key), var in ENV_OVERRIDES.items():
        if os.environ.get(var):
            cfg.setdefault(section, {})[key] = _coerce(os.environ[var], cfg.get(section, {}).get(key, ""))
    for section, values in (overrides or {}).items():
        cfg.setdefault(section, {}).update(values)

    for section in ("model", "preprocessing", "data", "gallery", "open_set"):
        if section not in cfg:
            raise ConfigError(f"Config {path} is missing required section '{section}'")
    for section, key in _PATH_KEYS:
        if key in cfg.get(section, {}):
            cfg[section][key] = resolve_path(cfg[section][key])
    cfg["_config_path"] = path
    return cfg


@dataclass(frozen=True)
class ReIDSettings:
    checkpoint: Path
    num_train_classes: int = 751
    embedding_dim: int = 512
    input_size: Tuple[int, int] = (256, 128)
    device: str = "auto"
    backend: str = "auto"
    onnx_path: Path = MODELS_DIR / "onnx" / "ps1_resnet18_embedding.onnx"
    tensorrt_dir: Path = MODELS_DIR / "tensorrt"
    tensorrt_precision: str = "fp32"
    tensorrt_auto_build: bool = True
    tensorrt_max_batch: int = 32
    tensorrt_opt_batch: int = 8
    verify_min_cosine: float = 0.9995
    batch_size: int = 32
    preprocessing: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def from_config(cls, cfg: dict) -> "ReIDSettings":
        m, p = cfg["model"], cfg["preprocessing"]
        defaults = cls(checkpoint=Path(m["checkpoint"]))
        pick = lambda k: m.get(k, getattr(defaults, k))
        return cls(
            checkpoint=Path(m["checkpoint"]), num_train_classes=int(pick("num_train_classes")),
            embedding_dim=int(pick("embedding_dim")), input_size=(int(p["height"]), int(p["width"])),
            device=str(pick("device")), backend=str(pick("backend")), onnx_path=Path(pick("onnx_path")),
            tensorrt_dir=Path(pick("tensorrt_dir")), tensorrt_precision=str(pick("tensorrt_precision")),
            tensorrt_auto_build=bool(pick("tensorrt_auto_build")), verify_min_cosine=float(pick("verify_min_cosine")),
            batch_size=int(pick("batch_size")),
            preprocessing=tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items()),
        )

    @property
    def preprocessing_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.preprocessing}

    @property
    def tensorrt_path(self) -> Path:
        return self.tensorrt_dir / f"{self.onnx_path.stem}_{self.tensorrt_precision}.engine"


@lru_cache(maxsize=16)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    st = Path(path).stat()
    return _sha256_cached(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)
