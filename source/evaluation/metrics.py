from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from source.logic.reid.open_set import threshold_for_far, threshold_for_tar


def _similarity_chunks(qf: np.ndarray, gf: np.ndarray, chunk: int = 512):
    for s in range(0, len(qf), chunk):
        yield s, qf[s:s + chunk] @ gf.T


def market_retrieval(qf, qp, qc, gf, gp, gc, max_rank: int = 10) -> dict:
    cmc = np.zeros(max_rank)
    aps, valid_q = [], 0
    for s, sim in _similarity_chunks(qf, gf):
        for r in range(sim.shape[0]):
            i = s + r
            order = np.argsort(-sim[r], kind="stable")
            keep = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
            matches = (gp[order] == qp[i])[keep]
            if not matches.any():
                continue
            valid_q += 1
            first = int(np.argmax(matches))
            if first < max_rank:
                cmc[first:] += 1
            hits = np.flatnonzero(matches)
            precision = (np.arange(len(hits)) + 1) / (hits + 1)
            aps.append(precision.mean())
    cmc /= max(valid_q, 1)
    return {"rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9]),
            "mAP": float(np.mean(aps)), "num_valid_queries": valid_q, "cmc": cmc.round(6).tolist()}


def pair_verification(qf, qp, qc, gf, gp, gc, n_pairs: int = 50000, seed: int = 0,
                      target_far: float = 0.01) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(seed)
    pos, neg = [], []
    by_pid = {p: np.flatnonzero((gp == p)) for p in np.unique(gp)}
    while len(pos) < n_pairs:
        i = rng.integers(len(qf))
        cand = by_pid.get(qp[i], np.array([], int))
        cand = cand[gc[cand] != qc[i]]
        if len(cand):
            pos.append(float(qf[i] @ gf[rng.choice(cand)]))
    while len(neg) < n_pairs:
        i, j = rng.integers(len(qf)), rng.integers(len(gf))
        if gp[j] != qp[i]:
            neg.append(float(qf[i] @ gf[j]))
    y = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    s = np.r_[pos, neg]
    fpr, tpr, thr = roc_curve(y, s)
    k = np.searchsorted(fpr, target_far, side="right") - 1
    return {"roc_auc": float(roc_auc_score(y, s)), "target_far": target_far,
            "threshold": float(thr[k]), "far": float(fpr[k]), "tar": float(tpr[k]),
            "positive_pairs": n_pairs, "negative_pairs": n_pairs, "seed": seed,
            "_curve": (fpr, tpr), "_scores": (np.asarray(pos), np.asarray(neg))}


@dataclass
class OpenSetSplit:
    known_q: np.ndarray
    unknown_q: np.ndarray
    gallery_mask: np.ndarray


def open_set_split(ids: np.ndarray, qp, gp, known_fraction: float, rng) -> OpenSetSplit:
    ids = rng.permutation(ids)
    n_known = int(round(len(ids) * known_fraction))
    known, unknown = set(ids[:n_known].tolist()), set(ids[n_known:].tolist())
    return OpenSetSplit(
        known_q=np.flatnonzero(np.isin(qp, list(known))),
        unknown_q=np.flatnonzero(np.isin(qp, list(unknown))),
        gallery_mask=np.isin(gp, list(known)),
    )


def open_set_scores(split: OpenSetSplit, qf, qp, qc, gf, gp, gc, aggregation: str = "max") -> dict:
    gidx = np.flatnonzero(split.gallery_mask)
    G, P, C = gf[gidx], gp[gidx], gc[gidx]
    uniq, inv = np.unique(P, return_inverse=True)
    if aggregation == "mean_prototype":
        proto_sum = np.zeros((len(uniq), G.shape[1]), np.float32)
        np.add.at(proto_sum, inv, G)
        protos = proto_sum / np.linalg.norm(proto_sum, axis=1, keepdims=True)
    elif aggregation != "max":
        raise ValueError(f"Unknown aggregation '{aggregation}'")

    out = {}
    for name, qidx in (("known", split.known_q), ("unknown", split.unknown_q)):
        top1_id = np.empty(len(qidx), dtype=gp.dtype)
        s1 = np.empty(len(qidx), np.float32)
        s2 = np.empty(len(qidx), np.float32)
        for start, sim in _similarity_chunks(qf[qidx], G):
            for r in range(sim.shape[0]):
                i = qidx[start + r]
                junk = (P == qp[i]) & (C == qc[i])
                if aggregation == "max":
                    row = np.where(junk, -np.inf, sim[r])
                    per_id = np.full(len(uniq), -np.inf, np.float32)
                    np.maximum.at(per_id, inv, row)
                else:
                    per_id = protos @ qf[i]
                    if junk.any():
                        j = int(np.searchsorted(uniq, qp[i]))
                        rest = proto_sum[j] - G[junk].sum(0)
                        nrm = float(np.linalg.norm(rest))
                        per_id[j] = rest @ qf[i] / nrm if nrm > 1e-6 else -np.inf
                top2 = np.argpartition(-per_id, 1)[:2]
                top2 = top2[np.argsort(-per_id[top2])]
                top1_id[start + r] = uniq[top2[0]]
                s1[start + r], s2[start + r] = per_id[top2[0]], per_id[top2[1]]
        out[name] = {"true_pid": qp[qidx], "top1_pid": top1_id, "s1": s1, "s2": s2}
    return out


def open_set_metrics(scores: dict, threshold: float, margin: float = 0.0) -> dict:
    def accept(d):
        return (d["s1"] >= threshold) & ((d["s1"] - d["s2"]) >= margin)

    k, u = scores["known"], scores["unknown"]
    ak, au = accept(k), accept(u)
    correct = k["top1_pid"] == k["true_pid"]
    n_k, n_u = len(k["s1"]), len(u["s1"])
    return {
        "threshold": float(threshold), "margin": float(margin),
        "DIR": float((ak & correct).mean()),
        "known_misidentified_accept_rate": float((ak & ~correct).mean()),
        "known_false_reject_rate": float((~ak).mean()),
        "unknown_FAR": float(au.mean()),
        "unknown_correct_reject_rate": float((~au).mean()),
        "overall_accuracy": float(((ak & correct).sum() + (~au).sum()) / (n_k + n_u)),
        "closed_set_rank1_on_enrolled": float(correct.mean()),
        "n_known_queries": n_k, "n_unknown_queries": n_u,
    }


def calibrate_threshold(cal_scores: dict, target_far: float, margin: float = 0.0) -> float:
    u = cal_scores["unknown"]
    effective = np.where((u["s1"] - u["s2"]) >= margin, u["s1"], -np.inf)
    return float(np.clip(threshold_for_far(effective, target_far), -1.0, 1.0))


def dir_far_curve(scores: dict, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k, u = scores["known"], scores["unknown"]
    correct = k["top1_pid"] == k["true_pid"]
    dirs = np.array([((k["s1"] >= t) & correct).mean() for t in thresholds])
    fars = np.array([(u["s1"] >= t).mean() for t in thresholds])
    return fars, dirs


def calibrate_threshold_for_tar(cal_scores: dict, target_tar: float, margin: float = 0.0) -> dict:
    k = cal_scores["known"]
    ok = (k["top1_pid"] == k["true_pid"]) & ((k["s1"] - k["s2"]) >= margin)
    genuine = np.where(ok, k["s1"], -np.inf)
    tau, attainable = threshold_for_tar(genuine, target_tar)
    return {"threshold": float(np.clip(tau, -1.0, 1.0)), "attainable": attainable,
            "n_calibration_known": int(len(genuine)), "max_attainable_tar": float(ok.mean()) if len(ok) else 0.0}


def all_pairs_verification(f: np.ndarray, pids: np.ndarray, camids: np.ndarray,
                           fars=(1e-3, 1e-2, 1e-1), cross_camera_genuine: bool = True,
                           impostor_mask: np.ndarray | None = None) -> dict:
    from sklearn.metrics import roc_auc_score

    f = np.asarray(f, np.float32)
    iu = np.triu_indices(len(f), k=1)
    sim = (f @ f.T)[iu]
    same = (pids[:, None] == pids[None, :])[iu]
    gen = same & ((camids[:, None] != camids[None, :])[iu] if cross_camera_genuine else True)
    imp = ~same if impostor_mask is None else (~same & np.asarray(impostor_mask, bool)[iu])
    g, i = sim[gen], sim[imp]
    out = {"genuine_pairs": int(len(g)), "impostor_pairs": int(len(i)), "operating_points": {}}
    if len(g) == 0 or len(i) == 0:
        out["error"] = "no genuine or no impostor pairs"
        return out
    out["roc_auc"] = float(roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(i))], np.r_[g, i]))
    for far in fars:
        tau = threshold_for_far(i, far)
        out["operating_points"][str(far)] = {
            "threshold": tau, "achieved_far": float((i >= tau).mean()), "tar": float((g >= tau).mean()),
            "resolvable": bool(len(i) * far >= 1)}
    out["_scores"] = (g, i)
    return out


def tar_far_at(scores: dict, threshold: float, margin: float = 0.0) -> dict:
    m = open_set_metrics(scores, threshold, margin)
    return {"threshold": float(threshold), "margin": float(margin), "TAR": m["DIR"], "FAR": m["unknown_FAR"],
            "misID": m["known_misidentified_accept_rate"]}
