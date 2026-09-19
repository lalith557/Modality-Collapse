"""E2 - Causal scale experiment (H2).

Sweeps the feature-magnitude ratio alpha_A/alpha_V over a fixed information
structure, with multiple seeds, and asks:

    "Does increasing scale disparity alone increase modality dominance while
     information (linear-probe accuracy) is held fixed?"

Usage:
    python experiments/e2_causal.py --config configs/synthetic_causal.yaml
    python experiments/e2_causal.py --config configs/synthetic_causal.yaml --seeds 3 --tag debug

Outputs (under results/):
    raw/e2_causal/<tag>/run_r{ratio}_s{seed}.json   per-run provenance + metrics
    processed/e2_causal_<tag>_runs.csv              flat per-run table
    processed/e2_causal_<tag>_agg.csv               aggregated over seeds
    tables/e2_causal_<tag>_summary.md               human-readable summary
    figures/e2_*.png                                plots
    raw/e2_causal/<tag>/run_record.json             experiment-level record
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.plots import make_e2_figures
from analysis.probes import linear_probe_accuracy
from analysis.statistics import aggregate_over_seeds, correlation, flatten_run
from data.synthetic import generate_dataset
from experiments.train import train_model
from msc.device import device_report, get_device
from msc.runlog import RunRecord, load_config


# --------------------------------------------------------------------------- #
# Automated quality control -- fail loudly on violated assumptions
# --------------------------------------------------------------------------- #
class QCError(RuntimeError):
    pass


def qc_check(runs: list[dict], probes: list[dict], tol_probe: float = 0.04,
             tol_scale: float = 0.12) -> list[str]:
    """Return a list of QC warnings; raise on hard violations."""
    warnings: list[str] = []
    df = pd.DataFrame(probes)

    # 1) Information invariance: within a seed, probe acc must be ~constant across ratios.
    for seed, g in df.groupby("seed"):
        for col in ("probe_acc_a", "probe_acc_v"):
            spread = float(g[col].max() - g[col].min())
            if spread > tol_probe:
                raise QCError(
                    f"[QC] probe {col} varies by {spread:.3f} across alpha within seed "
                    f"{seed} (> {tol_probe}). Information is NOT constant -> causal claim invalid."
                )

    # 2) Empirical scale ratio must match the intended alpha ratio.
    for p in probes:
        exp = p["alpha_ratio"]
        got = p["empirical_scale_ratio_test"]
        if exp > 0 and abs(got - exp) / exp > tol_scale:
            warnings.append(f"[QC] scale ratio mismatch: intended {exp}, empirical {got:.3f}")

    # 3) No NaN/None in core metrics.
    for r in runs:
        for k in ("final_rho", "early_rho"):
            if r[k] is None or (isinstance(r[k], float) and not np.isfinite(r[k])):
                warnings.append(f"[QC] non-finite {k} for ratio={r['alpha_ratio']} seed={r['seed']}")
    return warnings


def preliminary_h2_verdict(runs_df: pd.DataFrame, agg: list[dict]) -> dict:
    """Evidence-based, NOT hardcoded. Reports the numbers behind the call."""
    sp = correlation(list(runs_df["alpha_ratio"]), list(runs_df["test_dominance_a_minus_v"]))
    # collapse of the weak modality at the extremes vs balanced point
    dv_bal = runs_df.loc[np.isclose(runs_df["alpha_ratio"], 1.0), "test_delta_v"].mean()
    da_bal = runs_df.loc[np.isclose(runs_df["alpha_ratio"], 1.0), "test_delta_a"].mean()
    hi = runs_df.loc[runs_df["alpha_ratio"] >= 10, "test_delta_v"].mean()   # V weak when A huge
    lo = runs_df.loc[runs_df["alpha_ratio"] <= 0.1, "test_delta_a"].mean()  # A weak when V huge
    spearman = sp.get("spearman_r", sp.get("pearson_r", float("nan")))
    strong = (spearman is not None and spearman > 0.6
              and (dv_bal - hi) > 0.03 and (da_bal - lo) > 0.03)
    partial = (spearman is not None and spearman > 0.3)
    verdict = "SUPPORTED" if strong else ("PARTIAL" if partial else "INCONCLUSIVE")
    return {
        "verdict": verdict,
        "spearman_ratio_vs_dominance": spearman,
        "delta_v_at_balanced": float(dv_bal), "delta_v_at_ratio>=10": float(hi),
        "delta_a_at_balanced": float(da_bal), "delta_a_at_ratio<=0.1": float(lo),
        "note": "Preliminary automated read; the report interprets, not the code.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ratios", type=float, nargs="*", default=None)
    ap.add_argument("--margin", type=float, default=None, help="override prototype_margin")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.margin is not None:
        cfg["data"]["prototype_margin"] = args.margin
    seeds = cfg["seed_policy"]["seeds"]
    if args.seeds:
        seeds = list(range(args.seeds))
    ratios = args.ratios or cfg["sweep"]["ratios"]
    scale_mode = cfg["sweep"].get("scale_mode", "anchored_v")

    device = get_device(cfg["device"])
    print("Device:", device_report(device))
    print(f"Sweep ratios={ratios}  seeds={seeds}  epochs={cfg['train']['epochs']}  "
          f"scale_mode={scale_mode}  margin={cfg['data']['prototype_margin']}")

    out_root = Path(cfg["output"]["root"])
    if not out_root.is_absolute():
        out_root = ROOT / out_root       # anchor results to the repo, not CWD
    raw_dir = out_root / "raw" / "e2_causal" / args.tag
    raw_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    probes: list[dict] = []
    for ratio in ratios:
        for seed in seeds:
            params = dict(cfg["data"])
            params["data_seed"] = seed
            if scale_mode == "geometric":
                # symmetric: hold geometric-mean magnitude fixed, vary only the ratio
                params["alpha_a"] = float(ratio) ** 0.5
                params["alpha_v"] = float(ratio) ** -0.5
            elif scale_mode == "anchored_v":
                params["alpha_v"] = 1.0
                params["alpha_a"] = float(ratio)
            else:
                raise ValueError(f"unknown scale_mode '{scale_mode}'")
            ds = generate_dataset(params)

            # probe QC (GPU): information invariance check
            atr, vtr, ytr = ds.tensors["train"]
            ate, vte, yte = ds.tensors["test"]
            nc = ds.meta["n_classes"]
            na = ds.tensors["test"][0].norm(dim=1).mean()
            nv = ds.tensors["test"][1].norm(dim=1).mean()
            probes.append({
                "alpha_ratio": float(ratio), "seed": seed,
                "probe_acc_a": linear_probe_accuracy(atr, ytr, ate, yte, nc, device),
                "probe_acc_v": linear_probe_accuracy(vtr, ytr, vte, yte, nc, device),
                "empirical_scale_ratio_test": float(na / nv),
            })

            res = train_model(cfg, ds, device, seed=seed)
            # Use the intended ratio as the exact grouping key (avoids FP dust from
            # geometric alpha_a/alpha_v, e.g. 0.05 -> 0.04999999...).
            res["alpha_ratio"] = float(ratio)
            RunRecord(cfg, extra={"role": "e2_run"}).set(
                result={k: v for k, v in res.items() if k != "rho_history"},
                rho_history=res["rho_history"],
                device=device_report(device),
            ).save(raw_dir / f"run_r{ratio}_s{seed}.json")
            runs.append(res)
            print(f"  ratio={ratio:<6} seed={seed}  "
                  f"acc_av={res['test']['acc_av']:.3f}  "
                  f"delta_a={res['test']['delta_a']:+.3f} delta_v={res['test']['delta_v']:+.3f}  "
                  f"rho={res['final_rho']:.2f}")

    # QC
    warnings = qc_check(runs, probes)
    for w in warnings:
        print("WARNING", w)

    # Processed tables
    rows = [flatten_run(r) for r in runs]
    runs_df = pd.DataFrame(rows)
    proc = out_root / "processed"; proc.mkdir(parents=True, exist_ok=True)
    runs_csv = proc / f"e2_causal_{args.tag}_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    pd.DataFrame(probes).to_csv(proc / f"e2_causal_{args.tag}_probes.csv", index=False)

    agg = aggregate_over_seeds(rows, "alpha_ratio")
    pd.DataFrame(agg).to_csv(proc / f"e2_causal_{args.tag}_agg.csv", index=False)

    # Figures (namespaced by tag so runs don't overwrite each other)
    figs = make_e2_figures(agg, out_root / "figures" / args.tag)

    # H1 bonus signal: early rho vs final dominance (predictive?)
    h1 = correlation(list(runs_df["early_rho"]), list(runs_df["test_dominance_a_minus_v"]))

    verdict = preliminary_h2_verdict(runs_df, agg)

    # Summary table (markdown)
    tables = out_root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    summ = tables / f"e2_causal_{args.tag}_summary.md"
    with open(summ, "w", encoding="utf-8") as f:
        f.write(f"# E2 causal scale experiment ({args.tag})\n\n")
        f.write(f"seeds={seeds}, epochs={cfg['train']['epochs']}, fusion={cfg['model']['fusion']}, "
                f"norm={cfg['model']['normalization']}\n\n")
        f.write("| ratio | acc_av | delta_a | delta_v | gain | final_rho | early_rho | probe_a | probe_v |\n")
        f.write("|------:|-------:|--------:|--------:|-----:|----------:|----------:|--------:|--------:|\n")
        pdf = pd.DataFrame(probes)
        pr_map = pdf.groupby(pdf["alpha_ratio"].round(6)).mean(numeric_only=True)
        for a in agg:
            r = round(a["alpha_ratio"], 6)
            pa = pr_map.loc[r, "probe_acc_a"]; pv = pr_map.loc[r, "probe_acc_v"]
            f.write(f"| {r:g} | {a['test_acc_av_mean']:.3f} | {a['test_delta_a_mean']:+.3f} | "
                    f"{a['test_delta_v_mean']:+.3f} | {a['test_multimodal_gain_mean']:+.3f} | "
                    f"{a['final_rho_mean']:.2f} | {a['early_rho_mean']:.2f} | {pa:.3f} | {pv:.3f} |\n")
        f.write(f"\n**Preliminary H2 verdict:** {verdict['verdict']}\n\n")
        f.write(f"- Spearman(alpha_ratio, dominance) = {verdict['spearman_ratio_vs_dominance']:.3f}\n")
        f.write(f"- delta_v: balanced {verdict['delta_v_at_balanced']:+.3f} -> ratio>=10 {verdict['delta_v_at_ratio>=10']:+.3f}\n")
        f.write(f"- delta_a: balanced {verdict['delta_a_at_balanced']:+.3f} -> ratio<=0.1 {verdict['delta_a_at_ratio<=0.1']:+.3f}\n")
        f.write(f"- (bonus H1) corr(early_rho, dominance): pearson_r={h1.get('pearson_r')}, "
                f"spearman_r={h1.get('spearman_r')}\n")

    # Experiment-level record
    RunRecord(cfg, extra={"role": "e2_experiment", "tag": args.tag}).set(
        device=device_report(device), seeds=seeds, ratios=ratios, scale_mode=scale_mode,
        qc_warnings=warnings, h2_verdict=verdict, h1_bonus=h1,
        artifacts={"runs_csv": str(runs_csv), "figures": [str(p) for p in figs],
                   "summary": str(summ)},
    ).save(raw_dir / "run_record.json")

    print("\n===== E2 SUMMARY =====")
    print(pd.DataFrame(agg)[[
        "alpha_ratio", "test_acc_av_mean", "test_delta_a_mean",
        "test_delta_v_mean", "test_multimodal_gain_mean", "final_rho_mean", "early_rho_mean"
    ]].to_string(index=False))
    print("\nPreliminary H2 verdict:", verdict["verdict"], verdict)
    print("H1 bonus corr(early_rho, dominance):", h1)
    print("Figures:", [str(p) for p in figs])
    if warnings:
        print("QC warnings:", warnings)


if __name__ == "__main__":
    main()
