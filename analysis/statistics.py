"""Aggregation and simple statistics across seeds (mean, std, 95% CI)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as _sps
except Exception:                       # pragma: no cover
    _sps = None


def _ci95(vals: list[float]) -> tuple[float, float]:
    """Return (half_width, sd). Uses t-distribution when scipy is available."""
    a = np.asarray(vals, dtype=float)
    n = a.size
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return 0.0, sd
    se = sd / math.sqrt(n)
    if _sps is not None:
        t = float(_sps.t.ppf(0.975, df=n - 1))
    else:
        t = 1.96
    return t * se, sd


def flatten_run(run: dict) -> dict:
    """Flatten a train_model result into a single CSV-friendly row."""
    row = {
        "seed": run["seed"],
        "alpha_ratio": run["alpha_ratio"],
        "alpha_a": run["alpha_a"], "alpha_v": run["alpha_v"],
        "n_params": run["n_params"], "runtime_sec": run["runtime_sec"],
        "final_rho": run["final_rho"],
        "early_rho": run["early_rho"], "early_log_rho": run["early_log_rho"],
    }
    for k, v in run["test"].items():
        row[f"test_{k}"] = v
    return row


def aggregate_over_seeds(rows: list[dict], group_key: str = "alpha_ratio",
                         metrics: list[str] | None = None) -> list[dict]:
    """Group rows by ``group_key`` and compute mean/std/ci95 per metric."""
    if metrics is None:
        metrics = [k for k in rows[0] if k not in ("seed", group_key, "alpha_a", "alpha_v", "n_params")]
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)
    out = []
    for gval in sorted(groups):
        grp = groups[gval]
        agg = {group_key: gval, "n_seeds": len(grp)}
        for m in metrics:
            vals = [r[m] for r in grp if r.get(m) is not None]
            if not vals:
                continue
            ci, sd = _ci95(vals)
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = sd
            agg[f"{m}_ci95"] = ci
        out.append(agg)
    return out


def correlation(x: list[float], y: list[float]) -> dict:
    """Pearson and Spearman correlation (for the H1 early-rho signal)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    res = {}
    if _sps is not None and x.size >= 3:
        pr = _sps.pearsonr(x, y)
        sr = _sps.spearmanr(x, y)
        res["pearson_r"] = float(pr[0]); res["pearson_p"] = float(pr[1])
        res["spearman_r"] = float(sr[0]); res["spearman_p"] = float(sr[1])
    else:
        res["pearson_r"] = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 else float("nan")
    return res
