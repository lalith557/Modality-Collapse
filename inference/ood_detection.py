"""Feature-space distribution-shift detector (class-conditional Mahalanobis, tied covariance).

Protocol (integrity requirements):
  * reference statistics (class means, shared covariance) are fitted on TRAIN
    representations only;
  * the decision threshold is the 95th percentile of VALIDATION scores, i.e. it is
    chosen so that about 5% of in-distribution validation inputs are flagged;
  * the TEST split is never used to fit or tune -- only reported, evaluation-only.

Limitations: the threshold controls the false-alarm rate on in-distribution data. With
no labelled out-of-distribution data available, the detector's *detection* rate is
unknown, so a score below the threshold is NOT evidence that an input is in-distribution,
and a flag means only "potential distribution shift". The UTD-MHAD detector is fitted on
323 training clips (27 classes), so its covariance estimate is coarse.
"""
from __future__ import annotations

import numpy as np

QUANTILE = 0.95
SHRINK = 1e-3   # ridge as a fraction of the mean covariance diagonal, for invertibility


class MahalanobisDetector:
    def __init__(self, means: np.ndarray, precision: np.ndarray, threshold: float,
                 val_scores_q: dict | None = None):
        self.means = np.asarray(means, dtype=np.float64)          # [C, D]
        self.precision = np.asarray(precision, dtype=np.float64)  # [D, D]
        self.threshold = float(threshold)
        self.val_scores_q = val_scores_q or {}

    @classmethod
    def fit(cls, h_train: np.ndarray, y_train: np.ndarray, h_val: np.ndarray) -> "MahalanobisDetector":
        h = np.asarray(h_train, dtype=np.float64)
        y = np.asarray(y_train)
        classes = np.unique(y)
        means = np.stack([h[y == c].mean(axis=0) for c in classes])
        centred = h - means[np.searchsorted(classes, y)]
        cov = centred.T @ centred / len(h)
        cov += SHRINK * float(np.mean(np.diag(cov))) * np.eye(cov.shape[0])
        precision = np.linalg.inv(cov)
        det = cls(means, precision, threshold=np.inf)
        val = det.score(h_val)
        det.threshold = float(np.quantile(val, QUANTILE))
        det.val_scores_q = {f"q{int(q*100)}": float(np.quantile(val, q)) for q in (0.5, 0.9, 0.95, 0.99)}
        return det

    def score(self, h: np.ndarray) -> np.ndarray:
        """Minimum class-conditional squared Mahalanobis distance, per row."""
        h = np.atleast_2d(np.asarray(h, dtype=np.float64))
        diff = h[:, None, :] - self.means[None, :, :]                  # [N, C, D]
        d2 = np.einsum("ncd,de,nce->nc", diff, self.precision, diff)   # [N, C]
        return d2.min(axis=1)

    def assess(self, h: np.ndarray) -> dict:
        s = float(self.score(h)[0])
        return {"score": s, "threshold": self.threshold, "exceeds": bool(s > self.threshold),
                "ratio_to_threshold": s / self.threshold}

    def to_npz_dict(self, prefix: str) -> dict:
        return {f"{prefix}_means": self.means, f"{prefix}_precision": self.precision,
                f"{prefix}_threshold": np.array(self.threshold)}

    @classmethod
    def from_npz(cls, npz, prefix: str) -> "MahalanobisDetector":
        return cls(npz[f"{prefix}_means"], npz[f"{prefix}_precision"],
                   float(npz[f"{prefix}_threshold"]))
