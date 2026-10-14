from __future__ import annotations

import cv2
import numpy as np

H_BINS, V_BINS = 12, 4
DIM = 2 * (H_BINS + V_BINS)


def appearance_descriptor(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {rgb.shape}")
    hsv = cv2.cvtColor(cv2.resize(rgb, (64, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
    cols = slice(int(0.15 * 64), int(0.85 * 64))
    out = []
    for r0, r1 in ((0.15, 0.50), (0.50, 0.95)):
        reg = hsv[int(r0 * 128):int(r1 * 128), cols].reshape(-1, 3).astype(np.int32)
        h, s, v = reg[:, 0], reg[:, 1], reg[:, 2]
        chrom = (s > 40) & (v > 40)
        hh = np.bincount(np.minimum(h[chrom] * H_BINS // 180, H_BINS - 1), minlength=H_BINS)
        vv = np.bincount(np.minimum(v[~chrom] * V_BINS // 256, V_BINS - 1), minlength=V_BINS)
        hist = np.r_[hh, vv].astype(np.float64)
        out.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(out).astype(np.float32)


def identity_descriptors(desc: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids, inv = np.unique(pids, return_inverse=True)
    sums = np.zeros((len(ids), desc.shape[1]), np.float64)
    np.add.at(sums, inv, desc)
    mean = sums / np.bincount(inv)[:, None]
    return ids, (np.sqrt(mean) / np.sqrt(2.0)).astype(np.float32)


def lookalike_pairs(id_desc: np.ndarray, quantile: float = 0.02) -> tuple[np.ndarray, float]:
    sim = id_desc @ id_desc.T
    iu = np.triu_indices(len(sim), 1)
    cut = float(np.quantile(sim[iu], 1.0 - quantile))
    m = sim >= cut
    np.fill_diagonal(m, False)
    return m, cut
