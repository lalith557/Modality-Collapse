"""E1 - Early-rho diagnostic validation (H1). See PREREGISTRATION.md.

Central question: does log(rho at epoch 5) predict later modality dominance
BEYOND merely measuring the injected scale? We build a population that varies
seed, scale ratio, information difference (margin), noise difference (sigma),
encoder init (via seed) and fusion, then run the pre-registered beyond-scale
controls (partial correlation | injected scale; balanced-scale subfamily;
incremental R^2; per-scale stratification) and an AUROC-by-horizon curve.

Usage:
    python experiments/e1_diagnostic.py --config configs/diagnostic.yaml --tag primary
    python experiments/e1_diagnostic.py --config configs/diagnostic.yaml --tag debug --seeds 2 --epochs 12
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

from analysis.statistics import (auroc, auroc_ci, bootstrap_ci, correlation,
                                 incremental_r2, partial_correlation,
                                 stratified_correlation)
from data.synthetic import generate_dataset
from experiments.train import train_model
from msc.device import device_report, get_device
from msc.runlog import RunRecord, load_config


# --------------------------------------------------------------------------- #
# Build the varied run population
# --------------------------------------------------------------------------- #
def build_conditions(cfg: dict, seeds: list[int]) -> list[dict]:
    pop = cfg["population"]
    base = cfg["data"]
    conds: list[dict] = []

    def geo(scale):
        return scale ** 0.5, scale ** -0.5   # alpha_a, alpha_v

    # Family 'equal': information exactly equal; vary scale + seed + fusion.
    for scale in pop["scales_equal"]:
        aa, av = geo(scale)
        for fusion in pop["fusions"]:
            for seed in seeds:
                conds.append(dict(
                    family="equal", scale=float(scale), fusion=fusion, seed=seed,
                    margin_a=base["prototype_margin"], margin_v=base["prototype_margin"],
                    sigma_a=1.0, sigma_v=1.0, alpha_a=aa, alpha_v=av))

    # Family 'info_diff': scale=1, moderate margin asymmetry.
    for ma, mv in pop["info_margin_pairs"]:
        for fusion in pop["fusions"]:
            for seed in seeds:
                conds.append(dict(
                    family="info_diff", scale=1.0, fusion=fusion, seed=seed,
                    margin_a=float(ma), margin_v=float(mv),
                    sigma_a=1.0, sigma_v=1.0, alpha_a=1.0, alpha_v=1.0))

    # Family 'noise_diff': scale=1, moderate sigma asymmetry.
    for sa, sv in pop["noise_sigma_pairs"]:
        for fusion in pop["fusions"]:
            for seed in seeds:
                conds.append(dict(
                    family="noise_diff", scale=1.0, fusion=fusion, seed=seed,
                    margin_a=base["prototype_margin"], margin_v=base["prototype_margin"],
                    sigma_a=float(sa), sigma_v=float(sv), alpha_a=1.0, alpha_v=1.0))
    return conds


def dominance_by_epoch(history: list[dict]) -> dict[int, float]:
    return {int(r["epoch"]): r["val_delta_a"] - r["val_delta_v"] for r in history}


def rho_by_epoch(history: list[dict]) -> dict[int, float]:
    return {int(r["epoch"]): r["rho"] for r in history}


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze(runs_df: pd.DataFrame, dom_traj: list[dict], early_epoch: int, epochs: int) -> dict:
    out: dict = {"n_runs": int(len(runs_df))}
    x = runs_df["log_rho5"].values
    D = runs_df["D_final"].values
    A_strong = runs_df["A_stronger_final"].values.astype(int)
    log_scale = runs_df["log_scale"].values

    # (1) overall association
    out["overall_spearman"] = bootstrap_ci(x, D, "spearman")
    out["overall_pearson"] = bootstrap_ci(x, D, "pearson")
    out["auroc_final_A_stronger"] = auroc_ci(A_strong, x)

    # (2) beyond-scale controls
    out["partial_spearman_given_scale"] = partial_correlation(x, D, log_scale, "spearman")
    bal = runs_df["scale"] == 1.0
    out["balanced_subfamily_spearman"] = bootstrap_ci(
        runs_df.loc[bal, "log_rho5"].values, runs_df.loc[bal, "D_final"].values, "spearman")
    out["balanced_n"] = int(bal.sum())
    out["incremental_r2_rho5_over_scale"] = incremental_r2(D, log_scale, x)
    out["stratified_by_scale"] = stratified_correlation(runs_df, "log_rho5", "D_final", "scale", "spearman")

    # (3) per-family and balanced-per-family breakdown
    fam = {}
    for f, g in runs_df.groupby("family"):
        fam[f] = {
            "n": int(len(g)),
            "spearman_rho5_D": correlation(list(g["log_rho5"]), list(g["D_final"])).get("spearman_r"),
            "spearman_trueAsym_D": correlation(list(g["snr_a"] - g["snr_v"]), list(g["D_final"])).get("spearman_r"),
        }
    out["by_family"] = fam

    # (4) AUROC as a function of prediction horizon (epoch-5 rho -> A_stronger@epoch k)
    epochs_axis = list(range(1, epochs + 1))
    auroc_curve = []
    for e in epochs_axis:
        labels = np.array([1 if t.get(e, 0.0) > 0 else 0 for t in dom_traj], int)
        a = auroc(labels, x)
        # light bootstrap CI
        ci = auroc_ci(labels, x, n_boot=500, seed=e)
        auroc_curve.append({"epoch": e, "auroc": a, "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"]})
    out["auroc_by_epoch"] = auroc_curve

    # (5) sanity: does D reflect the true (snr) asymmetry at all?
    out["sanity_corr_trueAsym_D"] = correlation(
        list(runs_df["snr_a"] - runs_df["snr_v"]), list(runs_df["D_final"]))
    return out


def verdict(a: dict) -> dict:
    ov = abs(a["overall_spearman"]["point"])
    part = a["partial_spearman_given_scale"]
    bal = a["balanced_subfamily_spearman"]
    part_ok = abs(part["point"]) > 0.3 and part["excludes_zero"]
    bal_ok = abs(bal["point"]) > 0.3 and bal["excludes_zero"]
    # horizon decay: compare AUROC at epoch5 vs final
    curve = a["auroc_by_epoch"]
    au5 = next((c["auroc"] for c in curve if c["epoch"] == 5), float("nan"))
    au_final = curve[-1]["auroc"]
    decays = (au5 > 0.75) and (abs(au_final - 0.5) < abs(au5 - 0.5) - 0.15)

    if ov <= 0.3:
        v = "UNSUPPORTED"
    elif ov > 0.6 and part_ok and bal_ok:
        v = "SUPPORTED"
    elif ov > 0.6:
        v = "PARTIALLY SUPPORTED"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "overall_abs_spearman": ov,
            "partial_ok": part_ok, "balanced_ok": bal_ok,
            "auroc_epoch5": au5, "auroc_final": au_final, "horizon_decay": decays,
            "rule": "see PREREGISTRATION.md"}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_figures(runs_df, analysis, dom_traj, rho_traj, fig_dir: Path, early_epoch: int):
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"equal": "#1f77b4", "info_diff": "#2ca02c", "noise_diff": "#d62728"}

    # Fig 1: early rho vs final dominance, colored by family
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for f, g in runs_df.groupby("family"):
        ax.scatter(g["log_rho5"], g["D_final"], s=18, alpha=0.6,
                   c=colors.get(f, "gray"), label=f)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    sp = analysis["overall_spearman"]; pr = analysis["partial_spearman_given_scale"]
    bal = analysis["balanced_subfamily_spearman"]
    ax.set_xlabel(r"$\log(\rho_{\mathrm{epoch5}})$")
    ax.set_ylabel(r"final dominance $D=\Delta_A-\Delta_V$")
    ax.set_title("Early $\\rho$ vs final dominance")
    ax.text(0.02, 0.02,
            f"overall Spearman={sp['point']:.2f} [{sp['ci_lo']:.2f},{sp['ci_hi']:.2f}]\n"
            f"partial|scale={pr['point']:.2f} [{pr['ci_lo']:.2f},{pr['ci_hi']:.2f}]\n"
            f"balanced-only={bal['point']:.2f} [{bal['ci_lo']:.2f},{bal['ci_hi']:.2f}]",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "e1_early_rho_vs_final_dominance.png", dpi=150)
    plt.close(fig)

    # Fig 2: AUROC by prediction-horizon epoch
    curve = analysis["auroc_by_epoch"]
    ep = [c["epoch"] for c in curve]; au = [c["auroc"] for c in curve]
    lo = [c["ci_lo"] for c in curve]; hi = [c["ci_hi"] for c in curve]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(ep, au, color="#6a3d9a", marker="o", ms=3, label="AUROC (early $\\rho$ → A stronger @ epoch)")
    ax.fill_between(ep, lo, hi, color="#6a3d9a", alpha=0.2)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.axvline(early_epoch, color="0.5", ls=":", lw=1, label=f"early checkpoint (ep{early_epoch})")
    ax.set_xlabel("prediction-horizon epoch $k$"); ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Does epoch-5 $\\rho$ predict dominance at later epochs?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "e1_auroc_by_epoch.png", dpi=150)
    plt.close(fig)

    # Fig 3: rho trajectory examples (equal family, sum, seed 0) across scales
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for tr in rho_traj:
        if tr["family"] == "equal" and tr["fusion"] == "sum" and tr["seed"] == rho_traj[0]["seed"]:
            eps = sorted(tr["rho"].keys())
            ax.plot(eps, [math.log(tr["rho"][e]) for e in eps],
                    marker=".", ms=3, label=f"scale={tr['scale']:g}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(early_epoch, color="0.5", ls=":", lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\log \rho$ (fusion-boundary)")
    ax.set_title("Learned compensation: $\\log\\rho$ relaxes toward 0 during training")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "e1_rho_trajectory_examples.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seeds = cfg["seed_policy"]["seeds"]
    if args.seeds:
        seeds = list(range(args.seeds))
    epochs = int(cfg["train"]["epochs"])
    early_epoch = int(cfg["diagnostic"]["early_epoch"])

    device = get_device(cfg["device"])
    print("Device:", device_report(device))
    conds = build_conditions(cfg, seeds)
    print(f"E1 population: {len(conds)} runs (seeds={seeds}, epochs={epochs}, early={early_epoch})")

    out_root = ROOT / cfg["output"]["root"]
    raw_dir = out_root / "raw" / "e1_diagnostic" / args.tag
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, dom_traj, rho_traj = [], [], []
    for i, c in enumerate(conds):
        params = dict(cfg["data"])
        params.update(data_seed=c["seed"], alpha_a=c["alpha_a"], alpha_v=c["alpha_v"],
                      margin_a=c["margin_a"], margin_v=c["margin_v"],
                      sigma_a=c["sigma_a"], sigma_v=c["sigma_v"])
        ds = generate_dataset(params)
        cfg["model"]["fusion"] = c["fusion"]
        res = train_model(cfg, ds, device, seed=c["seed"])

        dom = dominance_by_epoch(res["rho_history"])
        rho = rho_by_epoch(res["rho_history"])
        rho5 = rho.get(early_epoch, res["final_rho"])
        Dfin = res["test"]["delta_a"] - res["test"]["delta_v"]
        row = {
            "family": c["family"], "fusion": c["fusion"], "seed": c["seed"],
            "scale": c["scale"], "log_scale": math.log(c["scale"]),
            "margin_a": c["margin_a"], "margin_v": c["margin_v"],
            "sigma_a": c["sigma_a"], "sigma_v": c["sigma_v"],
            "snr_a": ds.meta["snr_a"], "snr_v": ds.meta["snr_v"],
            "rho5": rho5, "log_rho5": math.log(rho5),
            "final_rho": res["final_rho"], "log_final_rho": math.log(res["final_rho"]),
            "D_final": Dfin, "A_stronger_final": int(Dfin > 0),
            "acc_av": res["test"]["acc_av"],
            "delta_a": res["test"]["delta_a"], "delta_v": res["test"]["delta_v"],
        }
        for e in (1, 3, 5, 10, 20, epochs):
            row[f"dom_e{e}"] = dom.get(e)
        rows.append(row)
        dom_traj.append(dom)
        rho_traj.append({"family": c["family"], "fusion": c["fusion"],
                         "seed": c["seed"], "scale": c["scale"], "rho": rho})
        if (i + 1) % 25 == 0 or i == len(conds) - 1:
            print(f"  [{i+1}/{len(conds)}] {c['family']:<10} scale={c['scale']:<5} "
                  f"{c['fusion']:<6} seed={c['seed']}  log_rho5={row['log_rho5']:+.2f}  D={Dfin:+.3f}")

    runs_df = pd.DataFrame(rows)
    proc = out_root / "processed"; proc.mkdir(parents=True, exist_ok=True)
    runs_csv = proc / f"e1_{args.tag}_runs.csv"
    runs_df.to_csv(runs_csv, index=False)

    analysis = analyze(runs_df, dom_traj, early_epoch, epochs)
    v = verdict(analysis)
    analysis["verdict"] = v

    figs = out_root / "figures" / "e1" / args.tag
    make_figures(runs_df, analysis, dom_traj, rho_traj, figs, early_epoch)
    # canonical figure names at results/figures/ for the deliverable list
    if args.tag == "primary":
        import shutil
        canon = out_root / "figures"
        for name in ["e1_early_rho_vs_final_dominance.png", "e1_auroc_by_epoch.png",
                     "e1_rho_trajectory_examples.png"]:
            shutil.copyfile(figs / name, canon / name)

    RunRecord(cfg, extra={"role": "e1_experiment", "tag": args.tag}).set(
        device=device_report(device), n_runs=len(conds), seeds=seeds,
        analysis=analysis).save(raw_dir / "analysis.json")

    # summary markdown
    tables = out_root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    with open(tables / f"e1_{args.tag}_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# E1 diagnostic ({args.tag}) — n={len(conds)} runs\n\n")
        sp = analysis["overall_spearman"]; pr = analysis["partial_spearman_given_scale"]
        bal = analysis["balanced_subfamily_spearman"]; ir = analysis["incremental_r2_rho5_over_scale"]
        f.write(f"**Verdict: {v['verdict']}**\n\n")
        f.write(f"- overall Spearman(log rho5, D_final) = {sp['point']:.3f} "
                f"[{sp['ci_lo']:.3f}, {sp['ci_hi']:.3f}], n={sp['n']}\n")
        f.write(f"- AUROC(final A_stronger | log rho5) = {analysis['auroc_final_A_stronger']['point']:.3f}\n")
        f.write(f"- **partial Spearman | log(scale) = {pr['point']:.3f} "
                f"[{pr['ci_lo']:.3f}, {pr['ci_hi']:.3f}]** (excl. 0: {pr['excludes_zero']})\n")
        f.write(f"- **balanced-scale subfamily Spearman = {bal['point']:.3f} "
                f"[{bal['ci_lo']:.3f}, {bal['ci_hi']:.3f}]** (n={analysis['balanced_n']})\n")
        f.write(f"- incremental R² of log rho5 over log(scale): "
                f"{ir['r2_base']:.3f} → {ir['r2_full']:.3f} (Δ={ir['increment']:.3f})\n")
        f.write(f"- AUROC decay: epoch5={v['auroc_epoch5']:.3f} → final={v['auroc_final']:.3f} "
                f"(decays={v['horizon_decay']})\n\n")
        f.write("### Per-family Spearman(log rho5, D_final) and (snr_a-snr_v, D)\n")
        for fam, d in analysis["by_family"].items():
            f.write(f"- {fam}: rho5→D = {d['spearman_rho5_D']:.3f}, "
                    f"trueAsym→D = {d['spearman_trueAsym_D']:.3f} (n={d['n']})\n")

    print("\n===== E1 SUMMARY =====")
    print("Verdict:", v["verdict"])
    print("overall Spearman:", round(analysis["overall_spearman"]["point"], 3),
          "partial|scale:", round(analysis["partial_spearman_given_scale"]["point"], 3),
          "balanced:", round(analysis["balanced_subfamily_spearman"]["point"], 3))
    print("AUROC epoch5→final:", round(v["auroc_epoch5"], 3), "→", round(v["auroc_final"], 3))
    print("by family:", {k: round(d["spearman_rho5_D"], 3) for k, d in analysis["by_family"].items()})
    print("figures:", str(figs))


if __name__ == "__main__":
    main()
