"""E4 - Scale x Learning-rate boundary. See PREREGISTRATION_E4.md.

Isolates scale x optimization (information held equal). Signed geometric grid over
scale ratio and per-encoder learning-rate ratio, comparing {none, zscore}. Tests
whether an unfavorable LR ratio blocks the learned compensation that otherwise
dissolves scale-driven collapse (E1/E3), producing PERSISTENT collapse.

Usage:
    python experiments/e4_boundary.py --config configs/boundary.yaml --tag smoke --seeds 2 --epochs 12 \
        --scales 0.1 1.0 10.0 --lrs 0.1 1.0 10.0
    python experiments/e4_boundary.py --config configs/boundary.yaml --tag primary
    python experiments/e4_boundary.py --config configs/boundary.yaml --tag concat_robust --fusion concat --seeds 3
"""
from __future__ import annotations

import argparse
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

from analysis.compensation import compensation_from_history
from analysis.statistics import bootstrap_ci
from data.synthetic import generate_dataset
from experiments.train import train_model, _optimizer
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import RunRecord, load_config


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
class QCError(RuntimeError):
    pass


def qc_optimizer_groups(cfg, device):
    """Verify per-encoder LR groups carry the intended learning rates."""
    c = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    c["train"] = dict(cfg["train"]); c["train"]["lr_a"] = 0.004; c["train"]["lr_v"] = 0.0001
    m = build_model(c, 32, 32, 8).to(device)
    opt = _optimizer(m, c, device)
    lrs = [g["lr"] for g in opt.param_groups]
    if not (abs(lrs[0] - 0.004) < 1e-9 and abs(lrs[1] - 0.0001) < 1e-9 and abs(lrs[2] - c["train"]["lr"]) < 1e-9):
        raise QCError(f"optimizer parameter groups wrong: {lrs}")
    return True


def qc_grid_complete(df, scales, lrs, norms, n_seeds):
    warnings = []
    for norm in norms:
        for s in scales:
            for l in lrs:
                g = df[(df["norm"] == norm) & (np.isclose(df["scale_r"], s)) & (np.isclose(df["lr_r"], l))]
                if len(g) != n_seeds:
                    warnings.append(f"cell(norm={norm},scale={s},lr={l}) has {len(g)}/{n_seeds} seeds")
    # symmetry of ratio sets
    for name, arr in (("scale", scales), ("lr", lrs)):
        for r in arr:
            if r != 1.0 and not any(np.isclose(1.0 / r, arr)):
                warnings.append(f"{name} ratio {r} has no reciprocal in the grid")
    if not np.isfinite(df["test_absdom"]).all():
        warnings.append("non-finite test imbalance present")
    return warnings


# --------------------------------------------------------------------------- #
# Run the grid
# --------------------------------------------------------------------------- #
def run_grid(cfg, device, scales, lrs, norms, seeds, epochs, final_window):
    base_lr = float(cfg["train"]["lr"])
    rows, window_traj, rho_traj = [], [], []
    total = len(norms) * len(scales) * len(lrs) * len(seeds)
    i = 0
    for norm in norms:
        for s in scales:
            aa, av = s ** 0.5, s ** -0.5
            for l in lrs:
                lr_a, lr_v = base_lr * (l ** 0.5), base_lr * (l ** -0.5)
                for seed in seeds:
                    i += 1
                    params = dict(cfg["data"])
                    params.update(data_seed=seed, alpha_a=aa, alpha_v=av)
                    # isolation QC: information must be equal
                    if params["sigma_a"] != params["sigma_v"] or \
                       params.get("margin_a", params["prototype_margin"]) != params.get("margin_v", params["prototype_margin"]):
                        raise QCError("information not equal in primary E4 cell")
                    ds = generate_dataset(params)
                    cfg["model"]["normalization"] = norm
                    cfg["train"]["lr_a"] = lr_a
                    cfg["train"]["lr_v"] = lr_v
                    res = train_model(cfg, ds, device, seed=seed)
                    hist = res["rho_history"]
                    if res["final_rho"] is None or not np.isfinite(res["final_rho"]):
                        raise QCError(f"non-finite rho at norm={norm} s={s} l={l} seed={seed}")

                    win_epochs = list(range(max(1, epochs - final_window + 1), epochs + 1))
                    win_min = {int(r["epoch"]): min(r["val_delta_a"], r["val_delta_v"])
                               for r in hist if int(r["epoch"]) in win_epochs}
                    weak_window = float(np.mean(list(win_min.values())))
                    absdom_window = float(np.mean([abs(r["val_delta_a"] - r["val_delta_v"])
                                                   for r in hist if int(r["epoch"]) in win_epochs]))
                    e5 = next((r for r in hist if int(r["epoch"]) == 5), None)
                    comp = compensation_from_history(hist, math.log(s), epochs)

                    row = {
                        "norm": norm, "scale_r": float(s), "lr_r": float(l),
                        "log2_scale": math.log2(s), "log2_lr": math.log2(l),
                        "seed": seed, "alpha_a": aa, "alpha_v": av, "lr_a": lr_a, "lr_v": lr_v,
                        "test_acc": res["test"]["acc_av"], "test_f1": res["test"]["f1_av"],
                        "test_delta_a": res["test"]["delta_a"], "test_delta_v": res["test"]["delta_v"],
                        "test_absdom": abs(res["test"]["delta_a"] - res["test"]["delta_v"]),
                        "test_min_delta": min(res["test"]["delta_a"], res["test"]["delta_v"]),
                        "test_gain": res["test"]["multimodal_gain"],
                        "weak_window": weak_window, "absdom_window": absdom_window,
                        "early_absdom": abs(e5["val_delta_a"] - e5["val_delta_v"]) if e5 else float("nan"),
                        "final_rho": res["final_rho"],
                        "rho_init": math.exp(comp["logrho_init"]),
                        "rho_ep5": math.exp(comp["logrho_ep5"]) if np.isfinite(comp["logrho_ep5"]) else float("nan"),
                        "rho_ep10": math.exp(comp["logrho_ep10"]) if np.isfinite(comp["logrho_ep10"]) else float("nan"),
                        "residual_abs_logrho": comp["residual_abs_logrho"],
                        "comp_applicable": comp["comp_applicable"],
                        "comp_ratio_final": comp.get("comp_ratio_final"),
                        "compensation_achieved": comp.get("compensation_achieved"),
                        "time_to_compensation": comp.get("time_to_compensation"),
                    }
                    rows.append(row)
                    window_traj.append({"norm": norm, "scale_r": float(s), "lr_r": float(l),
                                        "seed": seed, "win_min": win_min})
                    if i % 50 == 0 or i == total:
                        print(f"  [{i}/{total}] {norm:<7} scale={s:<5} lr={l:<5} seed={seed} "
                              f"weak={weak_window:+.3f} |D|={row['test_absdom']:.3f} residρ={row['residual_abs_logrho']:.2f}")
    return pd.DataFrame(rows), window_traj


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze(df, window_traj, cfg, scales, lrs, epochs, final_window):
    fw = cfg["e4"]
    halved = float(fw["collapse_halved_frac"]); stable = float(fw["collapse_stable_frac"])

    # baseline from (scale=1, lr=1, none)
    base = df[(df["norm"] == "none") & np.isclose(df["scale_r"], 1.0) & np.isclose(df["lr_r"], 1.0)]
    baseline_weak = float(base["weak_window"].mean())
    baseline_std = float(base["weak_window"].std(ddof=1))

    # per-epoch seed-mean weak in window, per (norm,scale,lr) for the stable-low check
    wt = pd.DataFrame([{**{k: t[k] for k in ("norm", "scale_r", "lr_r", "seed")},
                        "epoch": e, "mind": v}
                       for t in window_traj for e, v in t["win_min"].items()])
    cell_epoch_mean = wt.groupby(["norm", "scale_r", "lr_r", "epoch"])["mind"].mean().reset_index()

    def is_persistent(norm, s, l):
        weak_cell = df[(df["norm"] == norm) & np.isclose(df["scale_r"], s) & np.isclose(df["lr_r"], l)]["weak_window"].mean()
        cond1 = weak_cell <= halved * baseline_weak
        ce = cell_epoch_mean[(cell_epoch_mean["norm"] == norm) & np.isclose(cell_epoch_mean["scale_r"], s)
                             & np.isclose(cell_epoch_mean["lr_r"], l)]
        cond2 = bool((ce["mind"] <= stable * baseline_weak).all()) and len(ce) > 0
        severity = float(np.clip(1.0 - weak_cell / baseline_weak, 0, 1)) if baseline_weak > 0 else float("nan")
        return int(cond1 and cond2), severity, float(weak_cell)

    cells = []
    for norm in df["norm"].unique():
        for s in scales:
            for l in lrs:
                pc, sev, weak = is_persistent(norm, s, l)
                g = df[(df["norm"] == norm) & np.isclose(df["scale_r"], s) & np.isclose(df["lr_r"], l)]
                cells.append({
                    "norm": norm, "scale_r": s, "lr_r": l,
                    "log2_scale": math.log2(s), "log2_lr": math.log2(l),
                    "weak_window_mean": weak, "weak_window_std": float(g["weak_window"].std(ddof=1)),
                    "collapse_severity": sev, "persistent_collapse": pc,
                    "test_absdom_mean": float(g["test_absdom"].mean()),
                    "test_absdom_std": float(g["test_absdom"].std(ddof=1)),
                    "early_absdom_mean": float(g["early_absdom"].mean()),
                    "test_acc_mean": float(g["test_acc"].mean()),
                    "test_f1_mean": float(g["test_f1"].mean()),
                    "test_gain_mean": float(g["test_gain"].mean()),
                    "residual_abs_logrho_mean": float(g["residual_abs_logrho"].mean()),
                    "time_to_compensation_mean": float(g["time_to_compensation"].mean()) if g["comp_applicable"].any() else float("nan"),
                })
    cells_df = pd.DataFrame(cells)

    # paired normalization benefit per cell (weak zscore - none), CI
    stat_rows = []
    for s in scales:
        for l in lrs:
            n = df[(df["norm"] == "none") & np.isclose(df["scale_r"], s) & np.isclose(df["lr_r"], l)].sort_values("seed")
            z = df[(df["norm"] == "zscore") & np.isclose(df["scale_r"], s) & np.isclose(df["lr_r"], l)].sort_values("seed")
            if len(n) == 0 or len(z) == 0:
                continue
            diff = z["weak_window"].values - n["weak_window"].values
            m = float(np.mean(diff)); se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
            acc_diff = z["test_acc"].values - n["test_acc"].values
            stat_rows.append({
                "scale_r": s, "lr_r": l, "log2_scale": math.log2(s), "log2_lr": math.log2(l),
                "norm_benefit_weak_mean": m, "norm_benefit_weak_ci95": 1.96 * se,
                "norm_benefit_separated": bool(m - 1.96 * se > 0 or m + 1.96 * se < 0),
                "norm_benefit_acc_mean": float(np.mean(acc_diff)),
                "none_weak": float(n["weak_window"].mean()), "zscore_weak": float(z["weak_window"].mean()),
            })
    stats_df = pd.DataFrame(stat_rows)

    # boundary decomposition: severity (none) ~ |log2 scale|, |log2 lr|, alignment interaction
    none_cells = cells_df[cells_df["norm"] == "none"].copy()
    none_cells["abs_ls"] = none_cells["log2_scale"].abs()
    none_cells["abs_ll"] = none_cells["log2_lr"].abs()
    none_cells["align"] = np.sign(none_cells["log2_scale"]) * np.sign(none_cells["log2_lr"])
    none_cells["aligned_inter"] = none_cells["align"] * none_cells["abs_ls"] * none_cells["abs_ll"]
    boundary = _boundary_decomp(none_cells)
    boundary["mean_severity_by_alignment"] = {
        "aligned(+1)": float(none_cells.loc[none_cells["align"] > 0, "collapse_severity"].mean()),
        "orthogonal(0)": float(none_cells.loc[none_cells["align"] == 0, "collapse_severity"].mean()),
        "opposed(-1)": float(none_cells.loc[none_cells["align"] < 0, "collapse_severity"].mean()),
    }
    boundary["n_persistent_cells_none"] = int(none_cells["persistent_collapse"].sum())
    boundary["n_persistent_cells_zscore"] = int(cells_df[(cells_df["norm"] == "zscore")]["persistent_collapse"].sum())

    return {
        "baseline_weak": baseline_weak, "baseline_std": baseline_std,
        "cells": cells_df, "stats": stats_df, "boundary": boundary,
    }


def _boundary_decomp(nc):
    """R^2 of severity explained by scale magnitude, lr magnitude, and aligned interaction."""
    y = nc["collapse_severity"].fillna(0).values
    def r2(cols):
        X = np.column_stack([np.ones(len(y))] + [nc[c].values for c in cols])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = ((y - y.mean()) ** 2).sum()
        return float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "r2_scale_only": r2(["abs_ls"]),
        "r2_lr_only": r2(["abs_ll"]),
        "r2_scale_plus_lr": r2(["abs_ls", "abs_ll"]),
        "r2_plus_aligned_interaction": r2(["abs_ls", "abs_ll", "aligned_inter"]),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _grid_array(cells_df, norm, value, scales, lrs):
    Z = np.full((len(lrs), len(scales)), np.nan)
    for iy, l in enumerate(lrs):
        for ix, s in enumerate(scales):
            row = cells_df[(cells_df["norm"] == norm) & np.isclose(cells_df["scale_r"], s)
                           & np.isclose(cells_df["lr_r"], l)]
            if len(row):
                Z[iy, ix] = row[value].values[0]
    return Z


def _heatmap(ax, Z, scales, lrs, cmap, title, cbar_label, vmin=None, vmax=None, center=None):
    if center is not None:
        m = np.nanmax(np.abs(Z - center)) if np.isfinite(Z).any() else 1
        vmin, vmax = center - m, center + m
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(scales))); ax.set_xticklabels([f"{math.log2(s):.1f}" for s in scales], fontsize=7)
    ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f"{math.log2(l):.1f}" for l in lrs], fontsize=7)
    ax.set_xlabel(r"$\log_2(\alpha_A/\alpha_V)$  (scale)")
    ax.set_ylabel(r"$\log_2(lr_A/lr_V)$  (optimization)")
    ax.set_title(title, fontsize=10)
    ic = len(scales) // 2
    ax.axhline(ic, color="k", lw=0.6, alpha=0.4); ax.axvline(ic, color="k", lw=0.6, alpha=0.4)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(cbar_label, fontsize=8)
    return im


def make_figures(analysis, fig_dir, canon_dir, scales, lrs, tag):
    fig_dir.mkdir(parents=True, exist_ok=True)
    cells = analysis["cells"]

    def save(fig, name):
        fig.tight_layout(); fig.savefig(fig_dir / name, dpi=150)
        if tag == "primary":
            import shutil; canon_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fig_dir / name, canon_dir / name)
        plt.close(fig)

    # 1: imbalance
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    Z = _grid_array(cells, "none", "test_absdom_mean", scales, lrs)
    _heatmap(ax, Z, scales, lrs, "magma", "Final |Δ_A − Δ_V|  (none)", "imbalance", vmin=0)
    save(fig, "e4_phase_diagram_imbalance.png")

    # 2: collapse severity + PC markers
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    Z = _grid_array(cells, "none", "collapse_severity", scales, lrs)
    _heatmap(ax, Z, scales, lrs, "inferno", "Persistent-collapse severity  (none)", "severity S", vmin=0, vmax=1)
    PC = _grid_array(cells, "none", "persistent_collapse", scales, lrs)
    ys, xs = np.where(PC == 1)
    ax.scatter(xs, ys, marker="x", c="cyan", s=40, label="persistent collapse (PC=1)")
    if len(xs):
        ax.legend(fontsize=7, loc="upper left")
    save(fig, "e4_phase_diagram_collapse.png")

    # 3: normalization benefit (zscore weak - none weak)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    Zn = _grid_array(cells, "none", "weak_window_mean", scales, lrs)
    Zz = _grid_array(cells, "zscore", "weak_window_mean", scales, lrs)
    _heatmap(ax, Zz - Zn, scales, lrs, "RdBu", "Normalization benefit: weak(zscore) − weak(none)",
             "Δ weak-contribution", center=0.0)
    save(fig, "e4_normalization_benefit.png")

    # 4: compensation boundary (residual |log rho|, none)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    Z = _grid_array(cells, "none", "residual_abs_logrho_mean", scales, lrs)
    _heatmap(ax, Z, scales, lrs, "viridis", "Residual |log ρ| at convergence  (none)\n(high = compensation failed)",
             "residual |log ρ|", vmin=0)
    save(fig, "e4_compensation_boundary.png")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(path, analysis, cfg, scales, lrs, seeds, epochs, norms, fusion, qc_warnings, tag):
    b = analysis["boundary"]; cells = analysis["cells"]; stats = analysis["stats"]
    none = cells[cells["norm"] == "none"]
    bw = analysis["baseline_weak"]
    n_pc_none = b["n_persistent_cells_none"]; n_pc_z = b["n_persistent_cells_zscore"]
    # normalization recovery among none-collapsed cells
    recov = []
    for _, r in none[none["persistent_collapse"] == 1].iterrows():
        st = stats[np.isclose(stats["scale_r"], r["scale_r"]) & np.isclose(stats["lr_r"], r["lr_r"])]
        if len(st):
            recov.append(st["norm_benefit_weak_mean"].values[0])
    mean_recov = float(np.mean(recov)) if recov else float("nan")

    # outcome classification (data-driven)
    if n_pc_none == 0:
        outcome = "C — no persistent collapse even with LR disparity"
    elif mean_recov > 0.3 * bw:
        outcome = "A — persistent collapse appears AND z-score reliably recovers it"
    elif np.isfinite(mean_recov) and mean_recov <= 0.3 * bw:
        outcome = "B — persistent collapse appears but z-score does not reliably recover it"
    else:
        outcome = "D — pattern unclear; inspect"

    # boundary type
    inc = b["r2_plus_aligned_interaction"] - b["r2_scale_plus_lr"]
    if none["collapse_severity"].max() < 0.2:
        btype = "A (no boundary / negligible collapse anywhere)"
    elif inc > 0.1 and b["mean_severity_by_alignment"]["aligned(+1)"] > 2 * max(1e-6, b["mean_severity_by_alignment"]["opposed(-1)"]):
        btype = "C (clear alignment-driven boundary: collapse when scale & LR disfavour the same modality)"
    else:
        btype = "B (gradual transition)"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# E4 report — scale × learning-rate boundary ({tag})\n\n")
        f.write("## 1. Research question\n"
                "Does persistent modality collapse emerge when static feature-scale disparity is "
                "combined with asymmetric per-modality optimization (learning-rate) dynamics — i.e. "
                "can an unfavourable LR ratio block the learned compensation that otherwise dissolves "
                "scale-driven collapse by convergence?\n\n")
        f.write("## 2. Experimental design\n"
                f"Signed geometric grid: scale ratio {scales} × lr ratio {lrs} "
                f"(log2 axes, reciprocals included). Interventions {norms}. Fusion={fusion}. "
                f"{len(seeds)} seeds, {epochs} epochs. alpha_A=sqrt(r_s),alpha_V=1/sqrt(r_s); "
                f"lr_A=base*sqrt(r_l),lr_V=base/sqrt(r_l), base={cfg['train']['lr']}. "
                "Same generator/architecture/optimizer/batch as E2/E3.\n\n")
        f.write("## 3. Controls\nInformation held EXACTLY equal (margin_A=margin_V, sigma_A=sigma_V); "
                "only scale and per-encoder LR vary. QC: optimizer param-group LRs verified; grid "
                "completeness + ratio symmetry checked; no NaN; independent synthetic splits.\n")
        if qc_warnings:
            f.write("QC warnings: " + "; ".join(qc_warnings) + "\n")
        f.write("\n")
        f.write("## 4. Results\n")
        f.write(f"- Baseline weak-contribution (balanced cell, none) = {bw:.3f} ± {analysis['baseline_std']:.3f}.\n")
        f.write(f"- Persistent-collapse cells (none): **{n_pc_none}/{len(none)}**; (zscore): {n_pc_z}.\n")
        f.write(f"- Max collapse severity (none) = {none['collapse_severity'].max():.3f}; "
                f"mean = {none['collapse_severity'].mean():.3f}.\n")
        f.write(f"- Mean severity by alignment sign(log2 scale)*sign(log2 lr): "
                f"aligned={b['mean_severity_by_alignment']['aligned(+1)']:.3f}, "
                f"orthogonal={b['mean_severity_by_alignment']['orthogonal(0)']:.3f}, "
                f"opposed={b['mean_severity_by_alignment']['opposed(-1)']:.3f}.\n\n")
        f.write("## 5. Phase-diagram interpretation\n"
                f"Boundary type: **{btype}**. R²(severity): scale-only={b['r2_scale_only']:.2f}, "
                f"lr-only={b['r2_lr_only']:.2f}, scale+lr={b['r2_scale_plus_lr']:.2f}, "
                f"+aligned-interaction={b['r2_plus_aligned_interaction']:.2f} (Δ={inc:.2f}).\n"
                "See results/figures/e4_phase_diagram_imbalance.png and e4_phase_diagram_collapse.png.\n\n")
        f.write("## 6. Compensation analysis\n"
                f"Residual |log ρ| (none) ranges {none['residual_abs_logrho_mean'].min():.2f}–"
                f"{none['residual_abs_logrho_mean'].max():.2f}. Correlation(residual |log ρ|, collapse "
                f"severity) = {none['residual_abs_logrho_mean'].corr(none['collapse_severity']):.2f} "
                "— tests whether persistent collapse coincides with failed compensation. "
                "See e4_compensation_boundary.png.\n\n")
        f.write("## 7. Normalization effect\n")
        sep = int(stats["norm_benefit_separated"].sum()) if len(stats) else 0
        f.write(f"- z-score weak-contribution benefit vs none: {stats['norm_benefit_weak_mean'].mean():+.3f} "
                f"mean across cells; CI-separated in {sep}/{len(stats)} cells.\n")
        f.write(f"- Mean recovery within none-collapsed cells: {mean_recov:+.3f} "
                f"(baseline weak {bw:.3f}).\n")
        f.write(f"- Mean final-accuracy change (zscore − none) across grid: "
                f"{stats['norm_benefit_acc_mean'].mean():+.3f}.\n"
                "See e4_normalization_benefit.png.\n\n")
        f.write("## 8. Failures / caveats\n"
                "Only sum fusion + z-score in the primary grid; concat is a separate robustness run. "
                "40-epoch budget defines 'persistent' operationally; longer training could shift the "
                "boundary. z-score uses init-time train statistics (E3 caveat).\n\n")
        f.write("## 9. Statistical uncertainty\n"
                f"{len(seeds)} seeds/cell; per-cell std and paired (norm) 95% CIs in "
                "results/processed/e4_statistics.csv. No best-seed reporting.\n\n")
        f.write("## 10. Scientific conclusion\n"
                f"**Outcome {outcome}.**\n\n")
        f.write("## 11. Implications for H1/H2/H3\n"
                "- H2: whether static scale is *sufficient* for persistent collapse is answered by the "
                "LR=1 column vs the off-axis cells.\n"
                "- H1: early ρ remains a transient signal; residual |log ρ| is the convergence-time analogue "
                "used here.\n"
                "- H3: normalization's value (dynamics-only vs genuine recovery) is decided by the "
                "normalization-benefit map.\n\n")
        f.write("## 12. Recommended next experiment\n"
                "If Outcome A/B: E5 — implement OGM-GE faithfully and compare against z-score in the "
                "optimization-driven region only. If Outcome C: the architecture is compensation-capable; "
                "reconsider whether real-data collapse is scale- or optimization-driven before CREMA-D.\n")
    return outcome, btype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--scales", type=float, nargs="*", default=None)
    ap.add_argument("--lrs", type=float, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.fusion:
        cfg["model"]["fusion"] = args.fusion
    seeds = cfg["seed_policy"]["seeds"] if not args.seeds else list(range(args.seeds))
    epochs = int(cfg["train"]["epochs"])
    scales = args.scales or cfg["e4"]["scale_ratios"]
    lrs = args.lrs or cfg["e4"]["lr_ratios"]
    norms = cfg["e4"]["interventions"]
    final_window = int(cfg["e4"]["final_window"])

    device = get_device(cfg["device"])
    print("Device:", device_report(device))
    qc_optimizer_groups(cfg, device)
    print(f"E4: {len(scales)}x{len(lrs)} grid x {norms} x {len(seeds)} seeds "
          f"(fusion={cfg['model']['fusion']}, epochs={epochs}) = "
          f"{len(scales)*len(lrs)*len(norms)*len(seeds)} runs")

    out_root = ROOT / cfg["output"]["root"]
    df, window_traj = run_grid(cfg, device, scales, lrs, norms, seeds, epochs, final_window)

    qc_warnings = qc_grid_complete(df, scales, lrs, norms, len(seeds))
    for w in qc_warnings:
        print("QC WARNING:", w)

    proc = out_root / "processed"; proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc / f"e4_grid{'' if args.tag=='primary' else '_'+args.tag}.csv", index=False)

    analysis = analyze(df, window_traj, cfg, scales, lrs, epochs, final_window)
    summ = analysis["cells"]
    summ.to_csv(proc / f"e4_summary{'' if args.tag=='primary' else '_'+args.tag}.csv", index=False)
    analysis["stats"].to_csv(proc / f"e4_statistics{'' if args.tag=='primary' else '_'+args.tag}.csv", index=False)

    fig_dir = out_root / "figures" / "e4" / args.tag
    make_figures(analysis, fig_dir, out_root / "figures", scales, lrs, args.tag)

    reports = out_root / "reports"
    outcome, btype = write_report(reports / (f"E4_REPORT.md" if args.tag == "primary" else f"E4_REPORT_{args.tag}.md"),
                                  analysis, cfg, scales, lrs, seeds, epochs, norms, cfg["model"]["fusion"],
                                  qc_warnings, args.tag)

    raw_dir = out_root / "raw" / "e4_boundary" / args.tag; raw_dir.mkdir(parents=True, exist_ok=True)
    RunRecord(cfg, extra={"role": "e4_experiment", "tag": args.tag}).set(
        device=device_report(device), seeds=seeds, scales=scales, lrs=lrs,
        baseline_weak=analysis["baseline_weak"], boundary=analysis["boundary"],
        outcome=outcome, boundary_type=btype, qc_warnings=qc_warnings).save(raw_dir / "analysis.json")

    print("\n===== E4 SUMMARY =====")
    print(f"baseline weak={analysis['baseline_weak']:.3f}")
    print(f"persistent-collapse cells (none)={analysis['boundary']['n_persistent_cells_none']}, "
          f"(zscore)={analysis['boundary']['n_persistent_cells_zscore']}")
    print(f"mean severity by alignment: {analysis['boundary']['mean_severity_by_alignment']}")
    print(f"boundary R2: {analysis['boundary']}")
    print("OUTCOME:", outcome)
    print("BOUNDARY TYPE:", btype)


if __name__ == "__main__":
    main()
