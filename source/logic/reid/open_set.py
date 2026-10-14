from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from source.logic.reid.gallery import Ranking

UNKNOWN = "UNKNOWN"


@dataclass
class MatchResult:
    decision: str
    identity: object
    top1_score: float
    top2_score: float | None
    margin: float | None
    threshold: float
    margin_threshold: float
    reason: str
    ranking: Ranking

    @property
    def is_known(self) -> bool:
        return self.decision == "KNOWN"

    def to_dict(self, top_k: int | None = None) -> dict:
        g = self.ranking.gallery
        n = len(self.ranking.image_indices) if top_k is None else top_k
        return {
            "decision": self.decision,
            "identity": _py(self.identity),
            "top1_score": self.top1_score,
            "top2_score": self.top2_score,
            "margin": self.margin,
            "threshold": self.threshold,
            "margin_threshold": self.margin_threshold,
            "reason": self.reason,
            "top_identities": [
                {"pid": _py(p), "score": float(s), "filename": str(g.filenames[i])}
                for p, s, i in zip(self.ranking.identity_ids[:n], self.ranking.identity_scores[:n],
                                   self.ranking.identity_best_index[:n])
            ],
            "top_images": [
                {"pid": _py(g.pids[i]), "camid": _py(g.camids[i]), "filename": str(g.filenames[i]),
                 "score": float(s)}
                for i, s in zip(self.ranking.image_indices[:n], self.ranking.image_scores[:n])
            ],
        }


def _py(x):
    return x.item() if hasattr(x, "item") else x


class OpenSetMatcher:
    def __init__(self, threshold: float, margin: float = 0.0):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be a cosine similarity in [-1, 1], got {threshold}")
        if margin < 0:
            raise ValueError("margin must be >= 0")
        self.threshold = float(threshold)
        self.margin = float(margin)

    @classmethod
    def from_config(cls, cfg: dict) -> "OpenSetMatcher":
        o = cfg["open_set"]
        return cls(float(o["threshold"]), float(o.get("margin", 0.0)))

    def decide(self, ranking: Ranking) -> MatchResult:
        if len(ranking.identity_ids) == 0:
            return MatchResult("UNKNOWN", UNKNOWN, float("nan"), None, None, self.threshold, self.margin,
                               "gallery is empty", ranking)
        s1 = float(ranking.identity_scores[0])
        s2 = float(ranking.identity_scores[1]) if len(ranking.identity_scores) > 1 else None
        gap = (s1 - s2) if s2 is not None else None
        if s1 < self.threshold:
            reason = f"top-1 score {s1:.4f} < threshold {self.threshold:.4f}"
            return MatchResult("UNKNOWN", UNKNOWN, s1, s2, gap, self.threshold, self.margin, reason, ranking)
        if self.margin > 0 and gap is not None and gap < self.margin:
            reason = f"top1-top2 margin {gap:.4f} < required {self.margin:.4f} (ambiguous)"
            return MatchResult("UNKNOWN", UNKNOWN, s1, s2, gap, self.threshold, self.margin, reason, ranking)
        reason = f"top-1 score {s1:.4f} >= threshold {self.threshold:.4f}"
        return MatchResult("KNOWN", ranking.identity_ids[0], s1, s2, gap, self.threshold, self.margin,
                           reason, ranking)


def threshold_for_far(unknown_scores: np.ndarray, target_far: float) -> float:
    arr = np.asarray(unknown_scores)
    dt = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.dtype(np.float64)
    s = np.sort(arr.astype(dt))[::-1]
    if len(s) == 0:
        raise ValueError("No unknown scores to calibrate on")
    k = int(np.floor(target_far * len(s)))
    if k >= len(s):
        return float(s[-1])
    return float(np.nextafter(s[k], dt.type(np.inf)))


def threshold_for_tar(genuine_scores: np.ndarray, target_tar: float) -> tuple[float, bool]:
    if not 0.0 < target_tar <= 1.0:
        raise ValueError(f"target_tar must be in (0, 1], got {target_tar}")
    s = np.sort(np.asarray(genuine_scores, np.float64))[::-1]
    if len(s) == 0:
        raise ValueError("No genuine scores to calibrate on")
    need = int(np.ceil(target_tar * len(s) - 1e-9))
    t = s[need - 1]
    if not np.isfinite(t):
        return -1.0, False
    return float(t), True
