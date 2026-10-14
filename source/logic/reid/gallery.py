from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from source.errors import ArtifactError

REQUIRED_KEYS = ("features", "pids", "camids", "filenames")
META_KEYS = ("model_sha256", "model_checkpoint", "image_root")


@dataclass
class Ranking:
    identity_ids: np.ndarray
    identity_scores: np.ndarray
    identity_best_index: np.ndarray
    image_indices: np.ndarray
    image_scores: np.ndarray
    gallery: "Gallery" = field(repr=False)


class Gallery:
    def __init__(self, features: np.ndarray, pids: np.ndarray, camids: np.ndarray | None = None,
                 filenames: np.ndarray | None = None, source: str = "memory", meta: dict | None = None):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ArtifactError(f"Gallery features must be 2-D, got shape {features.shape}")
        n = len(features)
        pids = np.asarray(pids)
        if len(pids) != n:
            raise ArtifactError(f"pids length {len(pids)} != features rows {n}")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        self.features = features / np.maximum(norms, 1e-12)
        self.pids = pids
        self.camids = np.asarray(camids) if camids is not None else np.full(n, -1)
        self.filenames = np.asarray(filenames) if filenames is not None else np.array([""] * n)
        self.source = source
        self.meta = {k: str(v) for k, v in (meta or {}).items() if v}
        self._proto = None

    @classmethod
    def from_npz(cls, path: str | Path) -> "Gallery":
        path = Path(path)
        if not path.is_file():
            raise ArtifactError(f"Embedding file not found: {path}")
        with np.load(path, allow_pickle=False) as d:
            missing = [k for k in ("features", "pids") if k not in d.files]
            if missing:
                raise ArtifactError(f"{path.name} is missing arrays {missing}; found {d.files}")
            return cls(d["features"], d["pids"],
                       d["camids"] if "camids" in d.files else None,
                       d["filenames"] if "filenames" in d.files else None, source=str(path),
                       meta={k: str(d[k]) for k in META_KEYS if k in d.files})

    def save_npz(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=self.features, pids=self.pids,
                            camids=self.camids, filenames=self.filenames,
                            **{k: np.asarray(v) for k, v in self.meta.items() if k in META_KEYS})

    def __len__(self) -> int:
        return len(self.features)

    @property
    def dim(self) -> int:
        return self.features.shape[1]

    @property
    def identities(self) -> np.ndarray:
        return np.unique(self.pids)

    def add(self, features: np.ndarray, pid, camids=None, filenames=None) -> None:
        features = np.atleast_2d(np.asarray(features, np.float32))
        if features.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {features.shape[1]} != gallery dim {self.dim}")
        features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        k = len(features)
        new_pids = np.array([pid] * k)
        if new_pids.dtype != self.pids.dtype and len(self.pids):
            new_pids, self.pids = new_pids.astype(str), self.pids.astype(str)
        self.features = np.concatenate([self.features, features])
        self.pids = np.concatenate([self.pids, new_pids])
        self.camids = np.concatenate([self.camids, np.asarray(camids if camids is not None else [-1] * k)])
        self.filenames = np.concatenate([self.filenames.astype(str),
                                         np.asarray(filenames if filenames is not None else [""] * k, str)])
        self._proto = None

    def remove_identity(self, pid) -> int:
        keep = self.pids != pid
        removed = int((~keep).sum())
        self.subset_inplace(keep)
        return removed

    def subset(self, mask: np.ndarray) -> "Gallery":
        return Gallery(self.features[mask], self.pids[mask], self.camids[mask], self.filenames[mask], self.source,
                       self.meta)

    def subset_inplace(self, mask: np.ndarray) -> None:
        self.features, self.pids = self.features[mask], self.pids[mask]
        self.camids, self.filenames = self.camids[mask], self.filenames[mask]
        self._proto = None

    def _prototype_index(self):
        if self._proto is None:
            ids, inv = np.unique(self.pids, return_inverse=True)
            sums = np.zeros((len(ids), self.dim), np.float32)
            np.add.at(sums, inv, self.features)
            self._proto = (ids, inv, sums, np.bincount(inv, minlength=len(ids)))
        return self._proto

    def search(self, query: np.ndarray, top_k: int = 10, exclude_mask: np.ndarray | None = None,
               aggregation: str = "max") -> Ranking:
        q = np.asarray(query, np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"Query dim {q.shape[0]} != gallery dim {self.dim}")
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self.features @ q
        valid = np.ones(len(self), bool) if exclude_mask is None else ~np.asarray(exclude_mask, bool)
        if not valid.any():
            empty = np.array([], dtype=np.int64)
            return Ranking(empty, np.array([], np.float32), empty, empty, np.array([], np.float32), self)

        idx_valid = np.flatnonzero(valid)
        s_valid = sims[idx_valid]
        order = idx_valid[np.argsort(-s_valid, kind="stable")]

        uniq, first = np.unique(self.pids[order], return_index=True)
        best_idx = order[first]
        if aggregation == "max":
            id_score = sims[best_idx]
        elif aggregation == "mean_prototype":
            all_ids, inv_all, sums, counts = self._prototype_index()
            excluded = np.flatnonzero(~valid)
            if len(excluded):
                sums, counts = sums.copy(), counts.copy()
                np.add.at(sums, inv_all[excluded], -self.features[excluded])
                np.subtract.at(counts, inv_all[excluded], 1)
            protos = sums[counts > 0]
            id_score = (protos @ q) / np.maximum(np.linalg.norm(protos, axis=1), 1e-12)
        else:
            raise ValueError(f"Unknown aggregation '{aggregation}'")
        id_order = np.argsort(-id_score, kind="stable")

        return Ranking(
            identity_ids=uniq[id_order][:max(top_k, 2)],
            identity_scores=id_score[id_order][:max(top_k, 2)],
            identity_best_index=best_idx[id_order][:max(top_k, 2)],
            image_indices=order[:top_k],
            image_scores=sims[order[:top_k]],
            gallery=self,
        )
