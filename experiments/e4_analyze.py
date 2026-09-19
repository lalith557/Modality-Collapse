"""E4 rigorous post-hoc analysis (reads the completed grids; does NOT retrain the
primary grid). Produces QC, machine-readable summaries, interaction statistics,
aligned/orthogonal/opposed analysis, compensation & normalization analysis,
symmetry, boundary estimation, and Figures A-D. See PREREGISTRATION_E4.md.

Usage: python experiments/e4_analyze.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

OUT = ROOT / "results"
E4 = OUT / "e4"; E4.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
SCALES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
LRS = SCALES


def alignment_class(ls, ll):
    if ls == 0 and ll == 0:
        return "balanced"
    if ls == 0 or ll == 0:
        return "orthogonal"
    return "aligned" if ls * ll > 0 else "opposed"


def load(tag=None):
    name = "e4_grid.csv" if tag is None else f"e4_grid_{tag}.csv"
    df = pd.read_csv(OUT / "processed" / name)
    df["log2_scale"] = np.log2(df.scale_r); df["log2_lr"] = np.log2(df.lr_r)
    return df


def enrich(df, baseline_weak):
    df = df.copy()
    df["severity"] = np.clip(1 - df.weak_window / baseline_weak, 0, 1)
    df["alignment_class"] = [alignment_class(ls, ll) for ls, ll in zip(df.log2_scale, df.log2_lr)]
    df["run_collapse"] = (df.weak_window <= 0.5 * baseline_weak).astype(int)
    return df


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), float("nan"), float("nan")
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ols(y, X, names):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    p = 2 * sps.t.sf(np.abs(t), dof)
    tcrit = sps.t.ppf(0.975, dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot
    return ({names[i]: {"coef": float(beta[i]), "se": float(se[i]), "t": float(t[i]),
                        "p": float(p[i]), "ci": [float(beta[i] - tcrit * se[i]), float(beta[i] + tcrit * se[i])]}
             for i in range(k)}, float(r2), int(n))


def interaction_models(df):
    d = df.copy()
    d["abs_ls"] = d.log2_scale.abs(); d["abs_ll"] = d.log2_lr.abs()
    d["ls_ll"] = d.log2_scale * d.log2_lr          # = aligned interaction (signed)
    d["is_z"] = (d.norm == "zscore").astype(float)
    y = d.severity.values
    ones = np.ones(len(d))
    # magnitude + alignment model with normalization interactions
    cols = {
        "intercept": ones, "abs_ls": d.abs_ls, "abs_ll": d.abs_ll, "ls_ll(align)": d.ls_ll,
        "norm(z)": d.is_z, "norm:abs_ls": d.is_z * d.abs_ls,
        "norm:abs_ll": d.is_z * d.abs_ll, "norm:ls_ll": d.is_z * d.ls_ll,
    }
    names = list(cols); X = np.column_stack([np.asarray(cols[n], float) for n in names])
    full, r2_full, n = ols(y, X, names)
    # none-only reduced model (mechanism on the response surface)
    dn = d[d.norm == "none"]
    Xn = np.column_stack([np.ones(len(dn)), dn.abs_ls, dn.abs_ll, dn.ls_ll])
    none_m, r2_none, nn = ols(dn.severity.values, Xn, ["intercept", "abs_ls", "abs_ll", "ls_ll(align)"])
    return {"full_model": {"coeffs": full, "r2": r2_full, "n": n},
            "none_only_model": {"coeffs": none_m, "r2": r2_none, "n": nn}}


def alignment_analysis(df):
    out = {}
    for norm in ["none", "zscore"]:
        g = df[df.norm == norm]
        out[norm] = {}
        for cls in ["balanced", "orthogonal", "aligned", "opposed"]:
            sub = g[g.alignment_class == cls]
            m, lo, hi = boot_ci(sub.severity.values)
            wm, wlo, whi = boot_ci(sub.weak_window.values)
            out[norm][cls] = {
                "n": int(len(sub)),
                "severity_mean": m, "severity_ci": [lo, hi],
                "weak_mean": wm, "weak_ci": [wlo, whi],
                "gain_mean": float(sub.test_gain.mean()),
                "collapse_fraction": float(sub.run_collapse.mean()),
            }
    return out


def compensation_analysis(df):
    d = df[(df.norm == "none") & (df.comp_applicable == True)].copy()
    r_sev = sps.spearmanr(d.residual_abs_logrho, d.severity)
    r_ttc = sps.spearmanr(d.time_to_compensation, d.severity)
    # among aligned cells only
    da = d[d.alignment_class == "aligned"]
    r_sev_a = sps.spearmanr(da.residual_abs_logrho, da.severity) if len(da) > 3 else (float("nan"), float("nan"))
    return {
        "spearman_residual_vs_severity": {"r": float(r_sev.correlation), "p": float(r_sev.pvalue), "n": int(len(d))},
        "spearman_ttc_vs_severity": {"r": float(r_ttc.correlation), "p": float(r_ttc.pvalue)},
        "spearman_residual_vs_severity_aligned_only": {"r": float(r_sev_a[0]), "p": float(r_sev_a[1]), "n": int(len(da))},
        "residual_by_alignment": {
            c: float(d[d.alignment_class == c].residual_abs_logrho.mean())
            for c in ["orthogonal", "aligned", "opposed"]},
    }


def normalization_analysis(df, baseline_weak):
    rows = []
    for s in SCALES:
        for l in LRS:
            n = df[(df.norm == "none") & (df.scale_r == s) & (df.lr_r == l)].sort_values("seed")
            z = df[(df.norm == "zscore") & (df.scale_r == s) & (df.lr_r == l)].sort_values("seed")
            if len(n) != len(z) or len(n) == 0:
                continue
            dw = z.weak_window.values - n.weak_window.values
            da = z.test_acc.values - n.test_acc.values
            de = z.early_absdom.values - n.early_absdom.values
            rows.append({"scale_r": s, "lr_r": l, "log2_scale": math.log2(s), "log2_lr": math.log2(l),
                         "align_class": alignment_class(math.log2(s), math.log2(l)),
                         "d_weak": float(dw.mean()), "d_weak_se": float(dw.std(ddof=1) / math.sqrt(len(dw))),
                         "d_acc": float(da.mean()), "d_early_absdom": float(de.mean()),
                         "none_weak": float(n.weak_window.mean()), "z_weak": float(z.weak_window.mean())})
    nb = pd.DataFrame(rows)
    # overall + in aligned cells + in cells where none collapsed (per-run)
    collapsed_cells = df[(df.norm == "none")].groupby(["scale_r", "lr_r"]).run_collapse.mean()
    collapsed_keys = set(collapsed_cells[collapsed_cells >= 0.5].index)
    nb["none_collapsed_cell"] = [(r.scale_r, r.lr_r) in collapsed_keys for r in nb.itertuples()]
    return {
        "mean_d_weak_overall": float(nb.d_weak.mean()),
        "mean_d_weak_aligned": float(nb[nb["align_class"] == "aligned"].d_weak.mean()),
        "mean_d_weak_in_collapsed_cells": float(nb[nb.none_collapsed_cell].d_weak.mean()) if nb.none_collapsed_cell.any() else float("nan"),
        "mean_d_acc_overall": float(nb.d_acc.mean()),
        "mean_d_acc_aligned": float(nb[nb["align_class"] == "aligned"].d_acc.mean()),
        "mean_d_early_absdom_aligned": float(nb[nb["align_class"] == "aligned"].d_early_absdom.mean()),
        "n_collapsed_cells_none": int(len(collapsed_keys)),
        "table": nb,
    }


def symmetry_analysis(df):
    g = df[df.norm == "none"]
    a_fav = g[g.log2_scale > 0].severity            # A up-scaled
    v_fav = g[g.log2_scale < 0].severity            # V up-scaled
    # aligned corners: (scale+,lr+) vs (scale-,lr-)
    corner_A = g[(g.log2_scale > 0) & (g.log2_lr > 0)].severity   # V collapses
    corner_V = g[(g.log2_scale < 0) & (g.log2_lr < 0)].severity   # A collapses
    mw = sps.mannwhitneyu(corner_A, corner_V, alternative="two-sided")
    return {
        "severity_A_favored_mean": float(a_fav.mean()), "severity_V_favored_mean": float(v_fav.mean()),
        "aligned_corner_Aup_mean": float(corner_A.mean()), "aligned_corner_Vup_mean": float(corner_V.mean()),
        "aligned_corner_diff": float(corner_A.mean() - corner_V.mean()),
        "mannwhitney_p_corner": float(mw.pvalue),
    }


def cell_grid(df, norm, value):
    Z = np.full((len(LRS), len(SCALES)), np.nan)
    for iy, l in enumerate(LRS):
        for ix, s in enumerate(SCALES):
            sub = df[(df.norm == norm) & (df.scale_r == s) & (df.lr_r == l)]
            if len(sub):
                Z[iy, ix] = sub[value].mean()
    return Z


def collapse_fraction_grid(df, norm):
    Z = np.full((len(LRS), len(SCALES)), np.nan)
    for iy, l in enumerate(LRS):
        for ix, s in enumerate(SCALES):
            sub = df[(df.norm == norm) & (df.scale_r == s) & (df.lr_r == l)]
            if len(sub):
                Z[iy, ix] = sub.run_collapse.mean()
    return Z


def paired_heatmaps(df, fname, title, value_fn, cbar, cmap="inferno", vmin=None, vmax=None):
    Zn = value_fn(df, "none"); Zz = value_fn(df, "zscore")
    if vmax is None:
        vmax = np.nanmax([np.nanmax(Zn), np.nanmax(Zz)])
    if vmin is None:
        vmin = np.nanmin([np.nanmin(Zn), np.nanmin(Zz)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, Z, nm in zip(axes, [Zn, Zz], ["none", "z-score"]):
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([f"{math.log2(s):.1f}" for s in SCALES], fontsize=7)
        ax.set_yticks(range(len(LRS))); ax.set_yticklabels([f"{math.log2(l):.1f}" for l in LRS], fontsize=7)
        ax.set_xlabel(r"$\log_2(\alpha_A/\alpha_V)$ (scale)"); ax.set_ylabel(r"$\log_2(lr_A/lr_V)$")
        ic = len(SCALES) // 2; ax.axhline(ic, color="w", lw=0.5, alpha=0.5); ax.axvline(ic, color="w", lw=0.5, alpha=0.5)
        ax.set_title(f"{nm}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)


def main():
    df = load()
    baseline_weak = float(df[(df.norm == "none") & (df.scale_r == 1.0) & (df.lr_r == 1.0)].weak_window.mean())
    baseline_std = float(df[(df.norm == "none") & (df.scale_r == 1.0) & (df.lr_r == 1.0)].weak_window.std(ddof=1))
    df = enrich(df, baseline_weak)

    # QC table
    cells = df.groupby(["norm", "scale_r", "lr_r"]).size()
    qc = {
        "total_runs": int(len(df)), "expected": 810,
        "cells": int(len(cells)), "seeds_per_cell_ok": bool((cells == 5).all()),
        "n_scales": df.scale_r.nunique(), "n_lrs": df.lr_r.nunique(), "n_seeds": df.seed.nunique(),
        "norms": sorted(df.norm.unique()),
        "nonfinite_are_only_scale1_comp": bool(
            (~np.isfinite(df.select_dtypes(include=[np.number]))).any(axis=1).pipe(lambda m: (df[m].scale_r == 1.0).all())),
        "lr_param_max_err": float((df.lr_a - 0.001 * np.sqrt(df.lr_r)).abs().max()),
        "scale_param_max_err": float((df.alpha_a - np.sqrt(df.scale_r)).abs().max()),
        "baseline_weak": baseline_weak, "baseline_std": baseline_std,
        "balanced_acc": float(df[(df.norm == "none") & (df.scale_r == 1) & (df.lr_r == 1)].test_acc.mean()),
    }
    pd.DataFrame([qc]).to_csv(E4 / "e4_qc.csv", index=False)

    # per-run machine-readable summary (required columns)
    per_run = df.rename(columns={
        "scale_r": "scale_ratio", "lr_r": "lr_ratio", "norm": "normalization",
        "test_delta_a": "delta_a", "test_delta_v": "delta_v", "weak_window": "weak_modality_contribution",
        "test_gain": "multimodal_gain", "test_acc": "accuracy", "test_f1": "macro_f1",
        "rho_ep5": "rho_early", "final_rho": "rho_final", "residual_abs_logrho": "compensation_residual_abs_logrho",
    })[["scale_ratio", "lr_ratio", "normalization", "seed", "run_collapse", "severity", "delta_a", "delta_v",
        "weak_modality_contribution", "multimodal_gain", "accuracy", "macro_f1", "rho_early", "rho_final",
        "compensation_residual_abs_logrho", "compensation_achieved", "alignment_class"]]
    per_run.to_csv(E4 / "e4_primary_summary.csv", index=False)

    # aggregated per cell per norm
    agg = df.groupby(["norm", "scale_r", "lr_r", "alignment_class"]).agg(
        weak_mean=("weak_window", "mean"), weak_std=("weak_window", "std"),
        severity_mean=("severity", "mean"), severity_std=("severity", "std"),
        collapse_frac=("run_collapse", "mean"), acc_mean=("test_acc", "mean"),
        f1_mean=("test_f1", "mean"), gain_mean=("test_gain", "mean"),
        absdom_mean=("test_absdom", "mean"), rho_final_mean=("final_rho", "mean"),
        residual_logrho_mean=("residual_abs_logrho", "mean"), n=("seed", "count")).reset_index()
    agg["weak_ci95"] = 1.96 * agg["weak_std"] / np.sqrt(agg["n"])
    agg.to_csv(E4 / "e4_primary_aggregated.csv", index=False)

    # analyses
    inter = interaction_models(df)
    align = alignment_analysis(df)
    comp = compensation_analysis(df)
    normz = normalization_analysis(df, baseline_weak)
    sym = symmetry_analysis(df)

    # concat robustness (if present)
    concat = None
    if (OUT / "processed" / "e4_grid_concat_robust.csv").exists():
        dc = enrich(load("concat_robust"),
                    float(load("concat_robust").query("norm=='none' & scale_r==1 & lr_r==1").weak_window.mean()))
        concat = {"n_runs": int(len(dc)),
                  "aligned_sev": float(dc[(dc.norm == "none") & (dc.alignment_class == "aligned")].severity.mean()),
                  "opposed_sev": float(dc[(dc.norm == "none") & (dc.alignment_class == "opposed")].severity.mean()),
                  "n_run_collapse_none": int(dc[dc.norm == "none"].run_collapse.sum())}

    # persistent-collapse cell counts (frozen per-cell def, from summary produced by e4_boundary)
    summ = pd.read_csv(OUT / "processed" / "e4_summary.csv")
    pc_none = int(summ[(summ.norm == "none")].persistent_collapse.sum())
    pc_z = int(summ[(summ.norm == "zscore")].persistent_collapse.sum())

    stats_out = {
        "qc": qc, "baseline_weak": baseline_weak,
        "persistent_collapse_cells": {"none": pc_none, "zscore": pc_z, "total_cells": 81},
        "interaction_models": inter, "alignment_analysis": align,
        "compensation_analysis": comp, "symmetry_analysis": sym,
        "normalization_summary": {k: v for k, v in normz.items() if k != "table"},
        "concat_robustness": concat,
        "run_collapse_fraction": {"none": float(df[df.norm == "none"].run_collapse.mean()),
                                  "zscore": float(df[df.norm == "zscore"].run_collapse.mean())},
        "max_severity_none": float(df[df.norm == "none"].severity.max()),
        "mean_severity_none": float(df[df.norm == "none"].severity.mean()),
    }
    with open(E4 / "e4_analysis.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2)
    normz["table"].to_csv(E4 / "e4_normalization_by_cell.csv", index=False)

    # Figures A-D (paired none|zscore, shared scale)
    paired_heatmaps(df, "e4_final_A_severity.png", "Figure A — persistent-collapse severity",
                    lambda d, n: cell_grid(d, n, "severity"), "severity S", "inferno", 0, 1)
    paired_heatmaps(df, "e4_final_B_collapse_fraction.png", "Figure B — fraction of seeds with run-collapse",
                    lambda d, n: collapse_fraction_grid(d, n), "collapse fraction", "inferno", 0, 1)
    wmax = df.weak_window.max()
    paired_heatmaps(df, "e4_final_C_weak_contribution.png", "Figure C — weak-modality contribution",
                    lambda d, n: cell_grid(d, n, "weak_window"), "min(Δ_A,Δ_V)", "viridis", 0, wmax)
    paired_heatmaps(df, "e4_final_D_multimodal_gain.png", "Figure D — final multimodal gain",
                    lambda d, n: cell_grid(d, n, "test_gain"), "gain", "viridis")

    # compensation vs severity scatter (none, aligned highlighted)
    dna = df[(df.norm == "none") & (df.comp_applicable == True)]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for cls, c in [("opposed", "#1f77b4"), ("orthogonal", "#7f7f7f"), ("aligned", "#d62728")]:
        s = dna[dna.alignment_class == cls]
        ax.scatter(s.residual_abs_logrho, s.severity, s=12, alpha=0.5, c=c, label=cls)
    ax.set_xlabel(r"residual $|\log\rho|$ at convergence (compensation failure)")
    ax.set_ylabel("collapse severity S")
    r = comp["spearman_residual_vs_severity"]
    ax.set_title(f"Compensation failure vs collapse (none)\nSpearman={r['r']:.2f}, p={r['p']:.1e}, n={r['n']}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "e4_final_compensation_vs_severity.png", dpi=150); plt.close(fig)

    print("QC:", qc["total_runs"], "runs, seeds_ok:", qc["seeds_per_cell_ok"])
    print("baseline_weak:", round(baseline_weak, 3))
    print("persistent-collapse cells none/zscore:", pc_none, "/", pc_z)
    print("run-collapse fraction none:", round(stats_out["run_collapse_fraction"]["none"], 3),
          "zscore:", round(stats_out["run_collapse_fraction"]["zscore"], 3))
    print("alignment severity none:", {k: round(align["none"][k]["severity_mean"], 3) for k in align["none"]})
    print("none-only model R2:", round(inter["none_only_model"]["r2"], 3),
          "align coef:", {k: round(v["coef"], 4) for k, v in inter["none_only_model"]["coeffs"].items()})
    print("compensation Spearman(residual,severity):", round(comp["spearman_residual_vs_severity"]["r"], 3))
    print("norm benefit d_weak overall/aligned/in-collapsed:",
          round(normz["mean_d_weak_overall"], 3), round(normz["mean_d_weak_aligned"], 3),
          round(normz["mean_d_weak_in_collapsed_cells"], 3))
    print("symmetry aligned corners Aup/Vup:", round(sym["aligned_corner_Aup_mean"], 3),
          round(sym["aligned_corner_Vup_mean"], 3), "MWU p:", round(sym["mannwhitney_p_corner"], 3))
    if concat:
        print("concat: aligned_sev", round(concat["aligned_sev"], 3), "opposed", round(concat["opposed_sev"], 3),
              "run_collapse none:", concat["n_run_collapse_none"])


if __name__ == "__main__":
    main()
