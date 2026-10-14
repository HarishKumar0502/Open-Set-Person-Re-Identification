from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

from source.config import CONFIG_DIR
from source.logic.preprocessing import load_image

SELECTION_SEED = 2026
SELECTION_COUNT = 100
SELECTION_MANIFEST = CONFIG_DIR / "splits" / "selection_identities_v1.json"
_NAME = re.compile(r"^(-?\d+)_c(\d+)s\d+_\d+_\d+(?:\.jpg)+$", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    path: Path
    pid: int
    camid: int


def parse_market_name(name: str) -> tuple[int, int]:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"Not a Market-1501 filename: {name}")
    return int(m.group(1)), int(m.group(2))


def list_split(directory: Path, drop_junk: bool = True) -> list[Sample]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Market-1501 folder not found: {directory}")
    out = []
    for p in sorted(directory.glob("*.jpg")):
        try:
            pid, cam = parse_market_name(p.name)
        except ValueError:
            continue
        if drop_junk and pid <= 0:
            continue
        out.append(Sample(p, pid, cam))
    return out


def split_identities(pids, val_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(np.asarray(pids))
    if not 0 <= val_count < len(ids):
        raise ValueError(f"val_count {val_count} must be in [0, {len(ids)})")
    perm = np.random.default_rng(seed).permutation(ids)
    return np.sort(perm[val_count:]), np.sort(perm[:val_count])


def derive_selection_identities(pool_pids, count: int = SELECTION_COUNT, seed: int = SELECTION_SEED) -> np.ndarray:
    ids = np.unique(np.asarray(pool_pids))
    return np.sort(np.random.default_rng(seed).permutation(ids)[:count])


def selection_identities(manifest: str | Path = SELECTION_MANIFEST) -> np.ndarray:
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    pids = np.sort(np.asarray(m["pids"], dtype=np.int64))
    digest = hashlib.sha256(",".join(map(str, pids.tolist())).encode()).hexdigest()
    if len(pids) != m["count"] or len(np.unique(pids)) != len(pids) or digest != m["sha256_of_comma_joined_pids"]:
        raise ValueError(f"Selection manifest {manifest} failed its integrity check")
    return pids


def calibration_test_halves(eval_ids, seed: int) -> tuple[np.random.Generator, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(eval_ids))
    cal, test = perm[: len(perm) // 2], perm[len(perm) // 2:]
    assert_disjoint(calibration=cal, test=test)
    return rng, cal, test


def identity_roles(test_pool_pids, train_pids, seeds: int, selection=None) -> dict:
    test_pool = np.unique(np.asarray(test_pool_pids))
    sel = selection_identities() if selection is None else np.unique(np.asarray(selection))
    missing = np.setdiff1d(sel, test_pool)
    if len(missing):
        raise ValueError(f"selection ids not in the test pool: {missing[:5]}")
    eval_ids = np.setdiff1d(test_pool, sel)
    assert_disjoint(train=train_pids, selection=sel, evaluation=eval_ids)
    per_seed = []
    for s in range(seeds):
        _, cal, test = calibration_test_halves(eval_ids, s)
        assert_disjoint(train=train_pids, selection=sel, calibration=cal, test=test)
        per_seed.append({"seed": s, "calibration_pids": np.sort(cal).tolist(), "test_pids": np.sort(test).tolist()})
    return {"selection_pids": sel.tolist(), "evaluation_pids": eval_ids.tolist(), "per_seed": per_seed}


def assert_disjoint(**groups) -> None:
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = set(np.asarray(groups[a]).tolist()) & set(np.asarray(groups[b]).tolist())
            if both:
                raise ValueError(f"Identity leakage: {a} and {b} share {len(both)} ids, e.g. {sorted(both)[:5]}")


def train_transform(pre_cfg: dict) -> T.Compose:
    h, w = int(pre_cfg["height"]), int(pre_cfg["width"])
    interp = getattr(T.InterpolationMode, str(pre_cfg.get("interpolation", "bilinear")).upper())
    return T.Compose([
        T.Resize((h, w), interpolation=interp),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop((h, w)),
        T.ToTensor(),
        T.Normalize(pre_cfg["mean"], pre_cfg["std"]),
    ])


class MarketDataset(Dataset):
    def __init__(self, samples: list[Sample], transform, label_map: dict[int, int] | None = None):
        self.samples, self.transform = samples, transform
        self.label_map = label_map
        self.pids = np.array([s.pid for s in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        label = self.label_map[s.pid] if self.label_map is not None else s.pid
        return self.transform(load_image(s.path)), label, s.camid


class PKSampler(Sampler[list[int]]):
    def __init__(self, pids, p: int = 16, k: int = 4, seed: int = 0):
        pids = np.asarray(pids)
        self.index = {pid: np.flatnonzero(pids == pid) for pid in np.unique(pids)}
        if len(self.index) < 2:
            raise ValueError("PK sampling needs at least 2 identities (no negatives otherwise)")
        self.p, self.k, self.seed, self.epoch = min(p, len(self.index)), k, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index) // self.p

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 100_003 + self.epoch)
        ids = rng.permutation(list(self.index))
        for b in range(len(self)):
            batch = []
            for pid in ids[b * self.p:(b + 1) * self.p]:
                rows = self.index[pid]
                batch.extend(rng.choice(rows, self.k, replace=len(rows) < self.k).tolist())
            yield batch


def collate(batch):
    x, y, c = zip(*batch)
    return torch.stack(x), torch.as_tensor(y), torch.as_tensor(c)
