"""E3 - Per-modality normalization intervention (H3) + compensation analysis.

See PREREGISTRATION.md. Compares {none, layernorm, zscore, batchnorm} across a
symmetric geometric scale sweep, measuring at epochs 1/3/5/10/final:
rho, Delta_A, Delta_V, |Delta_A-Delta_V|, multimodal gain, accuracy, macro-F1,
convergence speed. The primary question is whether normalization prevents the
EARLY optimization imbalance, not merely final accuracy.

Compensation analysis quantifies how the injected scale (log alpha_A/alpha_V) is
absorbed into the fusion-boundary rho over training, and whether normalization
removes the need for that learned compensation.

Usage:
    python experiments/e3_cure.py --config configs/normalization.yaml --tag primary
    python experiments/e3_cure.py --config configs/normalization.yaml --tag debug --seeds 2 --epochs 12
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

from analysis.statistics import bootstrap_ci
from data.synthetic import generate_dataset
from experiments.train import train_model
from msc.device import device_report, get_device
from msc.runlog import RunRecord, load_config

NORM_COLORS = {"none": "#d62728", "layernorm": "#1f77b4",
               "zscore": "#2ca02c", "batchnorm": "#9467bd"}


def _row_at(history, epoch):
    for r in history:
        if int(r["epoch"]) == epoch:
            return r
    return None


def compensation_metrics(history, log_scale, epochs):
    """Quantify how injected scale is absorbed into rho over training."""
    if abs(log_scale) < 1e-9:
        return {"comp_applicable": False}
    logrho = {int(r["epoch"]): math.log(r["rho"]) for r in history}
    init = logrho.get(0, logrho[min(logrho)])
    final = logrho[max(logrho)]
    # time-to-compensation: first epoch with |log rho| < 0.5|log scale|
    ttc = None
    for e in sorted(logrho):
        if abs(logrho[e]) < 0.5 * abs(log_scale):
            ttc = e
            break
    return {
        "comp_applicable": True,
        "logrho_init": init,
        "logrho_final": final,
        "comp_ratio_init": init / log_scale,      # ~1 = no compensation
        "comp_ratio_final": final / log_scale,    # ~0 = full compensation
        "compensation_achieved": 1.0 - abs(final) / abs(log_scale),
        "time_to_compensation": ttc if ttc is not None else (epochs + 1),
        "residual_abs_logrho": abs(final),
    }


def geo(scale):
    return scale ** 0.5, scale ** -0.5


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
    e3 = cfg["e3"]
    interventions = e3["interventions"]
    scales = e3["scales"]
    checkpoints = list(e3["checkpoints"]) + [epochs]

    device = get_device(cfg["device"])
    print("Device:", device_report(device))
    print(f"E3: norms={interventions} scales={scales} seeds={seeds} epochs={epochs}")

    out_root = ROOT / cfg["output"]["root"]
    raw_dir = out_root / "raw" / "e3_cure" / args.tag
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, traj = [], []
    for norm in interventions:
        for scale in scales:
            aa, av = geo(scale)
            for seed in seeds:
                params = dict(cfg["data"])
                params.update(data_seed=seed, alpha_a=aa, alpha_v=av)
                ds = generate_dataset(params)
                cfg["model"]["normalization"] = norm
                res = train_model(cfg, ds, device, seed=seed)
                hist = res["rho_history"]
                log_scale = math.log(scale)

                row = {"norm": norm, "scale": float(scale), "log_scale": log_scale, "seed": seed}
                for cp in checkpoints:
                    r = _row_at(hist, cp)
                    if r is None:
                        continue
                    dom = r["val_delta_a"] - r["val_delta_v"]
                    tag = "final" if cp == epochs else str(cp)
                    row[f"dom_cp{tag}"] = dom
                    row[f"absdom_cp{tag}"] = abs(dom)
                    row[f"rho_cp{tag}"] = r["rho"]
                    row[f"logrho_cp{tag}"] = math.log(r["rho"])
                    row[f"gain_cp{tag}"] = r["val_gain"]
                    row[f"acc_cp{tag}"] = r["val_acc_av"]
                    row[f"f1_cp{tag}"] = r.get("val_f1_av")
                # final test metrics
                row["test_acc"] = res["test"]["acc_av"]
                row["test_f1"] = res["test"]["f1_av"]
                row["test_absdom"] = abs(res["test"]["delta_a"] - res["test"]["delta_v"])
                row["test_gain"] = res["test"]["multimodal_gain"]
                # convergence speed: first epoch reaching 98% of final val acc
                final_acc = _row_at(hist, epochs)["val_acc_av"]
                conv = epochs
                for r in hist:
                    if r["val_acc_av"] >= 0.98 * final_acc:
                        conv = int(r["epoch"]); break
                row["convergence_epoch"] = conv
                row.update({f"comp_{k}": v for k, v in
                            compensation_metrics(hist, log_scale, epochs).items()})
                rows.append(row)
                traj.append({"norm": norm, "scale": float(scale), "seed": seed,
                             "logrho": {int(r["epoch"]): math.log(r["rho"]) for r in hist},
                             "acc": {int(r["epoch"]): r["val_acc_av"] for r in hist},
                             "absdom": {int(r["epoch"]): abs(r["val_delta_a"] - r["val_delta_v"]) for r in hist}})
        print(f"  done norm={norm}")

    df = pd.DataFrame(rows)
    proc = out_root / "processed"; proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc / f"e3_{args.tag}_runs.csv", index=False)

    # aggregate by (norm, scale)
    agg = df.groupby(["norm", "scale"]).mean(numeric_only=True).reset_index()
    agg.to_csv(proc / f"e3_{args.tag}_agg.csv", index=False)

    analysis = _analyze(df, epochs)
    make_figures(df, traj, agg, out_root / "figures" / "e3" / args.tag, epochs, interventions)
    if args.tag == "primary":
        import shutil
        canon = out_root / "figures"; canon.mkdir(parents=True, exist_ok=True)
        for name in ["e3_norm_vs_scale_contribution.png", "e3_norm_rho_trajectory.png",
                     "e3_norm_convergence.png", "e3_norm_effect_size.png", "e3_compensation.png"]:
            src = out_root / "figures" / "e3" / args.tag / name
            if src.exists():
                shutil.copyfile(src, canon / name)

    RunRecord(cfg, extra={"role": "e3_experiment", "tag": args.tag}).set(
        device=device_report(device), analysis=analysis).save(raw_dir / "analysis.json")
    _write_summary(out_root / "tables" / f"e3_{args.tag}_summary.md", analysis, interventions, epochs)

    print("\n===== E3 SUMMARY =====")
    for norm in interventions:
        d = analysis["by_norm"][norm]
        print(f"  {norm:<10} early|D|@extreme={d['early_absdom_extreme_mean']:.3f}  "
              f"final|D|@extreme={d['final_absdom_extreme_mean']:.3f}  "
              f"test_acc={d['test_acc_mean']:.3f}  test_f1={d['test_f1_mean']:.3f}  "
              f"resid|logrho|={d['residual_abs_logrho_mean']:.3f}  conv_ep={d['convergence_epoch_mean']:.1f}")
    print("\nH3 verdict:", analysis["verdict"])


def _extreme_mask(df):
    return df["scale"].isin([0.05, 0.1, 10.0, 20.0])


def _analyze(df, epochs):
    ext = df[_extreme_mask(df)]
    by_norm = {}
    none_ext = ext[ext["norm"] == "none"]
    for norm in df["norm"].unique():
        g = ext[ext["norm"] == norm]
        gc = g[g["comp_comp_applicable"] == True]
        by_norm[norm] = {
            "early_absdom_extreme_mean": float(g["absdom_cp5"].mean()),
            "early_absdom_extreme_std": float(g["absdom_cp5"].std(ddof=1)),
            "final_absdom_extreme_mean": float(g["absdom_cpfinal"].mean()),
            "test_acc_mean": float(g["test_acc"].mean()),
            "test_f1_mean": float(g["test_f1"].mean()),
            "residual_abs_logrho_mean": float(gc["comp_residual_abs_logrho"].mean()) if len(gc) else float("nan"),
            "compensation_achieved_mean": float(gc["comp_compensation_achieved"].mean()) if len(gc) else float("nan"),
            "time_to_compensation_mean": float(gc["comp_time_to_compensation"].mean()) if len(gc) else float("nan"),
            "convergence_epoch_mean": float(g["convergence_epoch"].mean()),
        }
    # effect vs none: reduction in early |D| at extreme scales (paired by scale,seed)
    effects = {}
    for norm in df["norm"].unique():
        if norm == "none":
            continue
        merged = none_ext.merge(ext[ext["norm"] == norm], on=["scale", "seed"], suffixes=("_none", "_n"))
        diff = (merged["absdom_cp5_none"] - merged["absdom_cp5_n"]).values
        ci = bootstrap_ci(diff, diff, "pearson") if False else None
        # simple mean + 95% CI via normal approx
        m = float(np.mean(diff)); se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
        effects[norm] = {"early_absdom_reduction_mean": m,
                         "ci95": 1.96 * se, "n_pairs": int(len(diff)),
                         "separated": bool(m - 1.96 * se > 0 or m + 1.96 * se < 0)}
    # final accuracy effect vs none (all scales)
    acc_effect = {}
    none_all = df[df["norm"] == "none"]
    for norm in df["norm"].unique():
        if norm == "none":
            continue
        merged = none_all.merge(df[df["norm"] == norm], on=["scale", "seed"], suffixes=("_none", "_n"))
        diff = (merged["test_acc_n"] - merged["test_acc_none"]).values
        m = float(np.mean(diff)); se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
        acc_effect[norm] = {"final_acc_gain_vs_none_mean": m, "ci95": 1.96 * se}

    verdict = _verdict(by_norm, effects, acc_effect)
    return {"by_norm": by_norm, "early_imbalance_effect_vs_none": effects,
            "final_acc_effect_vs_none": acc_effect, "verdict": verdict}


def _verdict(by_norm, effects, acc_effect):
    none_early = by_norm["none"]["early_absdom_extreme_mean"]
    out = {}
    for norm, eff in effects.items():
        prevents_early = eff["separated"] and eff["early_absdom_reduction_mean"] > 0.3 * none_early
        helps_final = acc_effect[norm]["final_acc_gain_vs_none_mean"] - acc_effect[norm]["ci95"] > 0.005
        removes_comp = by_norm[norm]["residual_abs_logrho_mean"] < 0.5 * by_norm["none"]["residual_abs_logrho_mean"] \
            if np.isfinite(by_norm[norm]["residual_abs_logrho_mean"]) else False
        if prevents_early:
            v = "CURES early imbalance"
        elif eff["separated"] and eff["early_absdom_reduction_mean"] > 0:
            v = "PARTIALLY reduces early imbalance"
        else:
            v = "no clear early-imbalance reduction"
        out[norm] = {"summary": v, "prevents_early_imbalance": bool(prevents_early),
                     "improves_final_acc": bool(helps_final),
                     "reduces_learned_compensation_need": bool(removes_comp)}
    return out


def _msem(g, col):
    v = g[col].dropna().values
    if len(v) == 0:
        return np.nan, 0.0
    return float(np.mean(v)), (1.96 * float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0)


def make_figures(df, traj, agg, fig_dir, epochs, interventions):
    fig_dir.mkdir(parents=True, exist_ok=True)
    scales = sorted(df["scale"].unique())

    # Fig 1: |D| vs scale per norm, at epoch5 and final
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (cp, title) in zip(axes, [("cp5", "epoch 5 (early)"), ("cpfinal", "final")]):
        for norm in interventions:
            xs, ms, es = [], [], []
            for s in scales:
                g = df[(df["norm"] == norm) & (df["scale"] == s)]
                m, e = _msem(g, f"absdom_{cp}")
                xs.append(s); ms.append(m); es.append(e)
            ax.errorbar(xs, ms, yerr=es, marker="o", capsize=2, color=NORM_COLORS.get(norm), label=norm)
        ax.set_xscale("log"); ax.axvline(1.0, color="0.6", ls=":")
        ax.set_xlabel(r"scale ratio $\alpha_A/\alpha_V$"); ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"contribution imbalance $|\Delta_A-\Delta_V|$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Normalization vs scale-driven contribution imbalance")
    fig.tight_layout(); fig.savefig(fig_dir / "e3_norm_vs_scale_contribution.png", dpi=150); plt.close(fig)

    # Fig 2: log rho trajectory at extreme scale (max scale) per norm
    s_ext = max(scales)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for norm in interventions:
        curves = [t["logrho"] for t in traj if t["norm"] == norm and t["scale"] == s_ext]
        if not curves:
            continue
        eps = sorted(curves[0].keys())
        mean = [np.mean([c[e] for c in curves]) for e in eps]
        ax.plot(eps, mean, color=NORM_COLORS.get(norm), marker=".", ms=3, label=norm)
    ax.axhline(math.log(s_ext), color="k", ls="--", lw=0.8, label=f"injected log-scale ({math.log(s_ext):.2f})")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\log \rho$")
    ax.set_title(f"Fusion-boundary $\\log\\rho$ trajectory (scale={s_ext:g})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "e3_norm_rho_trajectory.png", dpi=150); plt.close(fig)

    # Fig 3: convergence — val acc trajectory at extreme scale per norm
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for norm in interventions:
        curves = [t["acc"] for t in traj if t["norm"] == norm and t["scale"] == s_ext]
        if not curves:
            continue
        eps = sorted(curves[0].keys())
        mean = [np.mean([c[e] for c in curves]) for e in eps]
        ax.plot(eps, mean, color=NORM_COLORS.get(norm), marker=".", ms=3, label=norm)
    ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy (A+V)")
    ax.set_title(f"Convergence (scale={s_ext:g})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "e3_norm_convergence.png", dpi=150); plt.close(fig)

    # Fig 4: effect size — early |D| reduction vs none at extreme scales
    ext = df[_extreme_mask(df)]
    none_ext = ext[ext["norm"] == "none"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels, means, errs = [], [], []
    for norm in interventions:
        if norm == "none":
            continue
        merged = none_ext.merge(ext[ext["norm"] == norm], on=["scale", "seed"], suffixes=("_none", "_n"))
        diff = (merged["absdom_cp5_none"] - merged["absdom_cp5_n"]).values
        labels.append(norm); means.append(np.mean(diff))
        errs.append(1.96 * np.std(diff, ddof=1) / math.sqrt(len(diff)))
    ax.bar(labels, means, yerr=errs, capsize=4, color=[NORM_COLORS.get(l) for l in labels])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"early $|\Delta_A-\Delta_V|$ reduction vs none")
    ax.set_title("Effect of normalization on EARLY imbalance (epoch 5, extreme scales)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "e3_norm_effect_size.png", dpi=150); plt.close(fig)

    # Fig 5: compensation — |log rho| / |log scale| trajectory at extreme scale
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for norm in interventions:
        curves = [t["logrho"] for t in traj if t["norm"] == norm and t["scale"] == s_ext]
        if not curves:
            continue
        ls = math.log(s_ext)
        eps = sorted(curves[0].keys())
        mean = [np.mean([c[e] / ls for c in curves]) for e in eps]
        ax.plot(eps, mean, color=NORM_COLORS.get(norm), marker=".", ms=3, label=norm)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="no compensation (ρ=injected)")
    ax.axhline(0.0, color="0.6", lw=0.8, label="full compensation (ρ=1)")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"compensation ratio $\log\rho_t / \log(\mathrm{scale})$")
    ax.set_title(f"Learned compensation of injected scale (scale={s_ext:g})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "e3_compensation.png", dpi=150); plt.close(fig)


def _write_summary(path, analysis, interventions, epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# E3 normalization intervention (H3)\n\n")
        f.write("| norm | early\\|D\\|@extreme | final\\|D\\|@extreme | test_acc | test_f1 | resid\\|logρ\\| | conv_ep |\n")
        f.write("|------|---:|---:|---:|---:|---:|---:|\n")
        for norm in interventions:
            d = analysis["by_norm"][norm]
            f.write(f"| {norm} | {d['early_absdom_extreme_mean']:.3f} | {d['final_absdom_extreme_mean']:.3f} | "
                    f"{d['test_acc_mean']:.3f} | {d['test_f1_mean']:.3f} | {d['residual_abs_logrho_mean']:.3f} | "
                    f"{d['convergence_epoch_mean']:.1f} |\n")
        f.write("\n### Effect vs none (early |D| reduction @ extreme scales, epoch 5)\n")
        for norm, e in analysis["early_imbalance_effect_vs_none"].items():
            f.write(f"- {norm}: {e['early_absdom_reduction_mean']:+.3f} ± {e['ci95']:.3f} "
                    f"(CI-separated: {e['separated']}, n={e['n_pairs']})\n")
        f.write("\n### Final accuracy effect vs none\n")
        for norm, e in analysis["final_acc_effect_vs_none"].items():
            f.write(f"- {norm}: {e['final_acc_gain_vs_none_mean']:+.3f} ± {e['ci95']:.3f}\n")
        f.write("\n### H3 verdict per intervention\n")
        for norm, v in analysis["verdict"].items():
            f.write(f"- **{norm}**: {v['summary']} | prevents_early={v['prevents_early_imbalance']} | "
                    f"improves_final_acc={v['improves_final_acc']} | "
                    f"reduces_compensation_need={v['reduces_learned_compensation_need']}\n")


if __name__ == "__main__":
    main()
