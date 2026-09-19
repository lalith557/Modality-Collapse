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
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"),
                "note": "constant input"}
    if _sps is not None and x.size >= 3:
        pr = _sps.pearsonr(x, y)
        sr = _sps.spearmanr(x, y)
        res["pearson_r"] = float(pr[0]); res["pearson_p"] = float(pr[1])
        res["spearman_r"] = float(sr[0]); res["spearman_p"] = float(sr[1])
    else:
        res["pearson_r"] = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 else float("nan")
    return res


# --------------------------------------------------------------------------- #
# E1 statistical tools: bootstrap CIs, AUROC, partial / stratified correlation
# --------------------------------------------------------------------------- #
def _spearman(x, y):
    if _sps is not None:
        return float(_sps.spearmanr(x, y).correlation)
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_ci(x, y, kind: str = "spearman", n_boot: int = 3000, seed: int = 0):
    """Point estimate + 95% percentile bootstrap CI for a correlation."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    fn = _spearman if kind == "spearman" else _pearson
    point = fn(x, y)
    rng = np.random.default_rng(seed)
    n = x.size
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        boots[b] = fn(xi, yi) if np.std(xi) > 0 and np.std(yi) > 0 else np.nan
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(n),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def auroc(labels, scores) -> float:
    """AUROC of `scores` predicting binary `labels`. Falls back if sklearn absent."""
    labels = np.asarray(labels, int); scores = np.asarray(scores, float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # Mann-Whitney U based AUROC
        order = np.argsort(scores)
        ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(scores) + 1)
        pos = labels == 1
        n1, n0 = pos.sum(), (~pos).sum()
        if n1 == 0 or n0 == 0:
            return float("nan")
        return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(labels, scores, n_boot: int = 3000, seed: int = 0):
    labels = np.asarray(labels, int); scores = np.asarray(scores, float)
    point = auroc(labels, scores)
    rng = np.random.default_rng(seed)
    n = labels.size
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auroc(labels[idx], scores[idx])
        if np.isfinite(a):
            boots.append(a)
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan)
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(n)}


def _residualize(y, Z):
    """Return residuals of OLS y ~ [1, Z]."""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    X = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_correlation(x, y, z, kind: str = "spearman", n_boot: int = 3000, seed: int = 0):
    """corr(x, y | z): correlation of x,y residuals after removing z.

    For 'spearman' we rank-transform x,y,z first (Spearman partial correlation).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    if kind == "spearman":
        x = np.argsort(np.argsort(x)).astype(float)
        y = np.argsort(np.argsort(y)).astype(float)
        z = np.argsort(np.argsort(z)).astype(float)
    rx = _residualize(x, z)
    ry = _residualize(y, z)
    ci = bootstrap_ci(rx, ry, kind="pearson", n_boot=n_boot, seed=seed)
    ci["kind"] = f"partial_{kind}"
    return ci


def stratified_correlation(df, xcol: str, ycol: str, stratcol: str, kind: str = "spearman"):
    """Per-stratum correlations + a Fisher-z pooled estimate."""
    fn = _spearman if kind == "spearman" else _pearson
    per = {}
    zs, ws = [], []
    for sval, g in df.groupby(stratcol):
        if len(g) >= 4 and g[xcol].std() > 0 and g[ycol].std() > 0:
            r = fn(g[xcol].values, g[ycol].values)
            per[float(sval)] = {"r": r, "n": int(len(g))}
            r_clip = np.clip(r, -0.999, 0.999)
            zs.append(np.arctanh(r_clip)); ws.append(len(g) - 3)
    pooled = float(np.tanh(np.average(zs, weights=ws))) if zs else float("nan")
    return {"per_stratum": per, "pooled_fisher_z": pooled, "n_strata": len(per)}


def incremental_r2(y, base_cols_matrix, add_col_vector):
    """R² of OLS(y~base) vs OLS(y~base+add); returns both and the increment."""
    y = np.asarray(y, float)
    def _r2(X):
        X = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    base = np.asarray(base_cols_matrix, float)
    add = np.asarray(add_col_vector, float)[:, None]
    r2_base = _r2(base)
    r2_full = _r2(np.column_stack([base, add]))
    return {"r2_base": r2_base, "r2_full": r2_full, "increment": r2_full - r2_base}
