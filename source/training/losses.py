from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TripletStats:
    loss: torch.Tensor
    valid_anchors: int
    active_fraction: float
    mean_pos_dist: float
    mean_neg_dist: float


def pairwise_sq_dist(emb: torch.Tensor) -> torch.Tensor:
    e = F.normalize(emb, dim=1)
    return (2.0 - 2.0 * e @ e.t()).clamp_min(0.0)


def _masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return same & ~eye, ~same


def triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3,
                 mining: str = "batch_hard") -> TripletStats:
    if emb.ndim != 2 or len(emb) != len(labels):
        raise ValueError(f"emb {tuple(emb.shape)} and labels {tuple(labels.shape)} do not match")
    d = pairwise_sq_dist(emb)
    pos, neg = _masks(labels)
    valid = pos.any(1) & neg.any(1)
    zero = emb.sum() * 0.0
    if not valid.any():
        return TripletStats(zero, 0, 0.0, float("nan"), float("nan"))

    if mining == "batch_hard":
        hardest_pos = torch.where(pos, d, torch.full_like(d, -1.0)).max(1).values
        hardest_neg = torch.where(neg, d, torch.full_like(d, float("inf"))).min(1).values
        per = F.relu(hardest_pos - hardest_neg + margin)[valid]
        loss = per.mean()
        return TripletStats(loss, int(valid.sum()), float((per > 0).float().mean()),
                            hardest_pos[valid].mean().item(), hardest_neg[valid].mean().item())
    if mining == "batch_all":
        t = d[:, :, None] - d[:, None, :] + margin
        m = pos[:, :, None] & neg[:, None, :]
        vals = F.relu(t[m])
        active = vals > 1e-12
        loss = vals[active].mean() if active.any() else zero
        return TripletStats(loss, int(valid.sum()), float(active.float().mean()),
                            d[pos].mean().item(), d[neg].mean().item())
    raise ValueError(f"Unknown mining '{mining}' (batch_hard | batch_all)")


class CosFaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.scale, self.margin = float(scale), float(margin)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cos = F.normalize(feats, dim=1) @ F.normalize(self.weight, dim=1).t()
        if labels is None:
            return self.scale * cos
        onehot = F.one_hot(labels, cos.shape[1]).to(cos.dtype)
        return self.scale * (cos - self.margin * onehot)
