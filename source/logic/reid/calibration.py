from __future__ import annotations

import numpy as np


def open_set_events(scores: dict) -> tuple[np.ndarray, np.ndarray]:
    k, u = scores["known"], scores["unknown"]
    x = np.r_[np.c_[k["s1"], k["s1"] - k["s2"]], np.c_[u["s1"], u["s1"] - u["s2"]]].astype(np.float64)
    y = np.r_[(k["top1_pid"] == k["true_pid"]).astype(int), np.zeros(len(u["s1"]), int)]
    return x, y


class ScoreCalibrator:
    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown calibration method '{method}'")
        self.method, self.model = method, None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScoreCalibrator":
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration needs both positive and negative events")
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=1e4, max_iter=1000).fit(x, y)
        else:
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x[:, 0], y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        if self.method == "platt":
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x[:, 0])


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(acc - conf)
        rows.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": int(m.sum()),
                     "mean_predicted": conf, "observed_rate": acc})
    return {"ece": float(ece), "bins": rows}


def evaluate_calibration(cal_scores: dict, test_scores: dict, bins: int = 10) -> dict:
    xc, yc = open_set_events(cal_scores)
    xt, yt = open_set_events(test_scores)
    out = {"n_calibration": int(len(yc)), "n_test": int(len(yt)), "test_positive_rate": float(yt.mean())}
    raw = np.clip(xt[:, 0], 0, 1)
    out["raw_cosine_as_probability"] = {"brier": brier(raw, yt), **reliability(raw, yt, bins)}
    out["constant_base_rate"] = {"brier": brier(np.full(len(yt), yc.mean()), yt)}
    for method in ("platt", "isotonic"):
        p = ScoreCalibrator(method).fit(xc, yc).predict(xt)
        out[method] = {"brier": brier(p, yt), **reliability(p, yt, bins)}
    return out
