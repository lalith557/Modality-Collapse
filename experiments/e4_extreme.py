"""E4-EXTREME — targeted extreme-ratio + long-horizon test. See PREREGISTRATION_E4_EXTREME.md.

Resolves whether stronger aligned scale x LR asymmetry converts graded imbalance
into persistent collapse, or the encoder keeps compensating. Reuses the E4
generator/architecture/optimizer/metrics and the FROZEN collapse definition.
Targeted conditions (not a grid); aligned conditions also run at 100/200 epochs.

Usage:
    python experiments/e4_extreme.py --tag smoke --seeds 2 --short   # quick check
    python experiments/e4_extreme.py --tag primary
"""
from __future__ import annotations

import argparse
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

from analysis.compensation import compensation_from_history
from data.synthetic import generate_dataset
from experiments.train import train_model
from msc.device import device_report, get_device
from msc.runlog import RunRecord, load_config

BASE_LR = 0.001
FINAL_WINDOW = 5
HALVED, STABLE = 0.5, 0.6


def build_conditions(mags):
    conds = [("balanced", 1.0, 1.0)]
    for m in mags:
        conds += [("scale_only", m, 1.0), ("scale_only", 1.0 / m, 1.0)]
        conds += [("lr_only", 1.0, m), ("lr_only", 1.0, 1.0 / m)]
        conds += [("aligned", m, m), ("aligned", 1.0 / m, 1.0 / m)]
        conds += [("opposed", m, 1.0 / m), ("opposed", 1.0 / m, m)]
    return conds


def favored(rs, rl):
    ls, ll = math.log2(rs), math.log2(rl)
    key = ls if ls != 0 else ll
    return "A" if key > 0 else ("V" if key < 0 else "none")


def run_one(cfg, device, rs, rl, norm, seed, epochs):
    p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=rs ** 0.5, alpha_v=rs ** -0.5)
    ds = generate_dataset(p)
    cfg = json.loads(json.dumps(cfg))  # deep copy (plain types)
    cfg["model"]["normalization"] = norm
    cfg["train"]["epochs"] = epochs
    cfg["train"]["lr_a"] = BASE_LR * (rl ** 0.5)
    cfg["train"]["lr_v"] = BASE_LR * (rl ** -0.5)
    try:
        res = train_model(cfg, ds, device, seed=seed)
    except (FloatingPointError, ValueError) as e:
        return {"diverged": True, "err": str(e)[:80]}, None
    h = sorted(res["rho_history"], key=lambda r: r["epoch"])
    win = list(range(max(1, epochs - FINAL_WINDOW + 1), epochs + 1))
    win_min = {int(r["epoch"]): min(r["val_delta_a"], r["val_delta_v"]) for r in h if int(r["epoch"]) in win}
    weak_window = float(np.mean(list(win_min.values())))
    comp = compensation_from_history(h, math.log(rs), epochs)
    e5 = next((r for r in h if int(r["epoch"]) == 5), None)
    scal = {
        "diverged": False,
        "weak_window": weak_window, "win_min": win_min,
        "test_acc": res["test"]["acc_av"], "test_f1": res["test"]["f1_av"],
        "delta_a": res["test"]["delta_a"], "delta_v": res["test"]["delta_v"],
        "absdom": abs(res["test"]["delta_a"] - res["test"]["delta_v"]),
        "gain": res["test"]["multimodal_gain"],
        "rho_early": e5["rho"] if e5 else float("nan"), "rho_final": res["final_rho"],
        "residual_abs_logrho": comp["residual_abs_logrho"],
        "comp_applicable": comp["comp_applicable"],
        "compensation_achieved": comp.get("compensation_achieved"),
        "time_to_compensation": comp.get("time_to_compensation"),
    }
    traj = {"epoch": [int(r["epoch"]) for r in h],
            "weak": [min(r["val_delta_a"], r["val_delta_v"]) for r in h],
            "logrho": [math.log(r["rho"]) for r in h],
            "gain": [r["val_gain"] for r in h],
            "delta_a": [r["val_delta_a"] for r in h], "delta_v": [r["val_delta_v"] for r in h]}
    return scal, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--short", action="store_true", help="smoke: mags=[20], horizons=[40]")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    device = get_device(cfg["device"])
    print("Device:", device_report(device))
    seeds = list(range(args.seeds))
    mags = [20] if args.short else [20, 100, 400]
    conds = build_conditions(mags)
    norms = ["none", "zscore"]

    # Set L: aligned + reference conditions at long horizons
    long_conds = ([("balanced", 1.0, 1.0)] +
                  [("aligned", m, m) for m in mags] +
                  [("aligned", 1.0 / max(mags), 1.0 / max(mags))] +
                  [("scale_only", max(mags), 1.0), ("opposed", max(mags), 1.0 / max(mags))])
    long_horizons = [40] if args.short else [100, 200]

    rows, trajs = [], {}
    rid = 0

    def do(cls, rs, rl, norm, seed, epochs):
        nonlocal rid
        scal, traj = run_one(cfg, device, rs, rl, norm, seed, epochs)
        row = {"run_id": rid, "cls": cls, "r_s": rs, "r_l": rl,
               "log2_scale": math.log2(rs), "log2_lr": math.log2(rl),
               "favored": favored(rs, rl), "norm": norm, "seed": seed, "epochs": epochs}
        row.update({k: v for k, v in scal.items() if k != "win_min"})
        rows.append(row)
        if traj is not None:
            trajs[rid] = {"win_min": scal.get("win_min"), **traj,
                          "cls": cls, "r_s": rs, "r_l": rl, "norm": norm, "seed": seed, "epochs": epochs}
        rid += 1

    total = len(conds) * len(norms) * len(seeds) + len(long_conds) * len(norms) * len(seeds) * len(long_horizons)
    print(f"E4-EXTREME: {len(conds)} conds x {norms} x {len(seeds)} seeds @40ep + "
          f"{len(long_conds)} long conds x {len(long_horizons)} horizons = ~{total} runs")

    # Set M (40 epochs)
    for cls, rs, rl in conds:
        for norm in norms:
            for seed in seeds:
                do(cls, rs, rl, norm, seed, 40)
    # Set L (long horizons)
    for cls, rs, rl in long_conds:
        for h in long_horizons:
            for norm in norms:
                for seed in seeds:
                    do(cls, rs, rl, norm, seed, h)

    df = pd.DataFrame(rows)
    proc = ROOT / "results" / "processed"; proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc / f"e4_extreme_runs{'' if args.tag=='primary' else '_'+args.tag}.csv", index=False)
    n_div = int(df.diverged.sum())
    print(f"runs={len(df)}  diverged={n_div}")

    analysis = analyze(df, trajs, mags, long_horizons)
    E4 = ROOT / "results" / "e4"; E4.mkdir(parents=True, exist_ok=True)
    with open(E4 / f"e4_extreme_analysis{'' if args.tag=='primary' else '_'+args.tag}.json", "w") as f:
        json.dump(analysis, f, indent=2, default=float)

    make_figures(df, trajs, analysis, mags, ROOT / "results" / "figures", args.tag)

    RunRecord({}, extra={"role": "e4_extreme", "tag": args.tag}).set(
        device=device_report(device), n_runs=len(df), n_diverged=n_div,
        analysis=analysis).save(ROOT / "results" / "raw" / "e4_extreme" / args.tag / "analysis.json")

    print("\n===== E4-EXTREME =====")
    print("baseline_weak:", round(analysis["baseline_weak"], 3))
    print("aligned dose (none) severity by mag:", analysis["dose_response"]["aligned_none"]["severity_by_mag"])
    print("aligned dose (none) residual by mag:", analysis["dose_response"]["aligned_none"]["residual_by_mag"])
    print("PC conditions @40ep (none):", analysis["pc"]["n_pc_40_none"], "| survive long:", analysis["pc"]["pc_survive_long"])
    print("aligned 400 weak @40/100/200 (none):", analysis["long_horizon"].get("aligned_400_none_weak_by_h"))
    print("classification:", analysis["classification"])


# --------------------------------------------------------------------------- #
def _boot(x, n=4000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x), True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _cond_weak(df, cls, rs, rl, norm, epochs):
    g = df[(df.cls == cls) & np.isclose(df.r_s, rs) & np.isclose(df.r_l, rl)
           & (df.norm == norm) & (df.epochs == epochs) & (~df.diverged)]
    return g


def _pc(df, trajs, cls, rs, rl, norm, epochs, baseline):
    g = _cond_weak(df, cls, rs, rl, norm, epochs)
    if len(g) == 0:
        return 0, float("nan")
    weak = g.weak_window.mean()
    sev = float(np.clip(1 - weak / baseline, 0, 1))
    # stable-window: seed-mean per-epoch weak <= 0.6*baseline across final window
    win_epochs = list(range(max(1, epochs - FINAL_WINDOW + 1), epochs + 1))
    per_ep = {e: [] for e in win_epochs}
    for rid in g.run_id:
        wm = trajs.get(rid, {}).get("win_min") or {}
        for e in win_epochs:
            if e in wm:
                per_ep[e].append(wm[e])
    stable = all(np.mean(per_ep[e]) <= STABLE * baseline for e in win_epochs if per_ep[e])
    pc = int(weak <= HALVED * baseline and stable)
    return pc, sev


def analyze(df, trajs, mags, long_horizons):
    baseline = float(_cond_weak(df, "balanced", 1.0, 1.0, "none", 40).weak_window.mean())

    def _rs_rl(cls, m):
        if cls == "aligned":
            return m, m
        if cls == "scale_only":
            return m, 1.0
        if cls == "lr_only":
            return 1.0, m
        if cls == "opposed":
            return m, 1.0 / m
        raise ValueError(cls)

    def dose(cls, norm):
        sev, res, weak = {}, {}, {}
        for m in mags:
            rs, rl = _rs_rl(cls, m)
            g = _cond_weak(df, cls, rs, rl, norm, 40)
            w = g.weak_window.mean()
            sev[m] = float(np.clip(1 - w / baseline, 0, 1)); weak[m] = float(w)
            res[m] = float(g.residual_abs_logrho.mean()) if g.comp_applicable.any() else float("nan")
        return {"severity_by_mag": sev, "residual_by_mag": res, "weak_by_mag": weak}

    dose_response = {
        "aligned_none": dose("aligned", "none"), "aligned_zscore": dose("aligned", "zscore"),
        "scale_only_none": dose("scale_only", "none"), "lr_only_none": dose("lr_only", "none"),
    }
    # monotonicity of aligned-none severity & residual vs log2 magnitude
    from scipy import stats as sps
    xm = [math.log2(m) for m in mags]
    sev_series = [dose_response["aligned_none"]["severity_by_mag"][m] for m in mags]
    res_series = [dose_response["aligned_none"]["residual_by_mag"][m] for m in mags]
    mono = {"spearman_sev": float(sps.spearmanr(xm, sev_series).correlation),
            "spearman_res": float(sps.spearmanr(xm, res_series).correlation)}

    # persistent collapse census (40 ep) + long-horizon survival
    pc40 = {}
    for cls, rs, rl in build_conditions(mags):
        for norm in ["none", "zscore"]:
            pc, sev = _pc(df, trajs, cls, rs, rl, norm, 40, baseline)
            if pc:
                pc40[f"{cls}|{rs:g}|{rl:g}|{norm}"] = sev
    n_pc_40_none = sum(1 for k in pc40 if k.endswith("none"))

    long_horizon = {}
    survive = {}
    for m in mags:
        by_h = {}
        for h in [40] + long_horizons:
            g = _cond_weak(df, "aligned", m, m, "none", h)
            if len(g):
                by_h[h] = float(g.weak_window.mean())
        long_horizon[f"aligned_{m}_none_weak_by_h"] = by_h
        # survival: PC at the longest horizon
        hlong = max(long_horizons)
        pc_l, _ = _pc(df, trajs, "aligned", m, m, "none", hlong, baseline)
        survive[f"aligned_{m}_none_pc_at_{hlong}"] = pc_l
    pc_survive_long = int(sum(survive.values()))

    # normalization decomposition (z removes scale, leaves LR?) at 40 ep, largest mag
    M = max(mags)
    def znone(cls, rs, rl):
        n = _cond_weak(df, cls, rs, rl, "none", 40).weak_window.mean()
        z = _cond_weak(df, cls, rs, rl, "zscore", 40).weak_window.mean()
        return {"none_weak": float(n), "zscore_weak": float(z), "d_weak": float(z - n)}
    norm_decomp = {
        "scale_only": znone("scale_only", M, 1.0),
        "lr_only": znone("lr_only", 1.0, M),
        "aligned": znone("aligned", M, M),
        "opposed": znone("opposed", M, 1.0 / M),
    }

    # classification
    n_div = int(df.diverged.sum())
    aligned400 = long_horizon.get(f"aligned_{max(mags)}_none_weak_by_h", {})
    recovers = (aligned400.get(max(long_horizons), 0) > aligned400.get(40, 1) + 0.02) if aligned400 else False
    if n_div > 0.1 * len(df):
        cls = "EXTREME-D — UNSTABLE/INCONCLUSIVE"
    elif pc_survive_long >= 1 and n_pc_40_none >= 1:
        cls = "EXTREME-A — TRUE PERSISTENT REGIME"
    elif mono["spearman_res"] > 0.6 and (n_pc_40_none >= 1 or dose_response["aligned_none"]["severity_by_mag"][max(mags)] > 0.4):
        cls = "EXTREME-B — GRADED COMPENSATION FAILURE"
    elif dose_response["aligned_none"]["severity_by_mag"][max(mags)] < 0.2:
        cls = "EXTREME-C — COMPENSATION ROBUST"
    else:
        cls = "EXTREME-B — GRADED COMPENSATION FAILURE"

    return {
        "baseline_weak": baseline, "n_runs": int(len(df)), "n_diverged": n_div,
        "dose_response": dose_response, "monotonicity": mono,
        "pc": {"pc40": pc40, "n_pc_40_none": n_pc_40_none, "survive": survive, "pc_survive_long": pc_survive_long},
        "long_horizon": {**long_horizon, "aligned_max_recovers_by_long": bool(recovers)},
        "normalization_decomposition": norm_decomp,
        "classification": cls,
    }


def make_figures(df, trajs, analysis, mags, figdir, tag):
    figdir.mkdir(parents=True, exist_ok=True)
    sfx = "" if tag == "primary" else f"_{tag}"

    # 1) dose-response: severity + residual vs aligned magnitude (none vs zscore)
    xm = [math.log2(m) for m in mags]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    dr = analysis["dose_response"]
    axes[0].plot(xm, [dr["aligned_none"]["severity_by_mag"][m] for m in mags], "o-", color="#d62728", label="aligned (none)")
    axes[0].plot(xm, [dr["aligned_zscore"]["severity_by_mag"][m] for m in mags], "s--", color="#9467bd", label="aligned (z-score)")
    axes[0].plot(xm, [dr["scale_only_none"]["severity_by_mag"][m] for m in mags], "^:", color="#1f77b4", label="scale-only (none)")
    axes[0].plot(xm, [dr["lr_only_none"]["severity_by_mag"][m] for m in mags], "v:", color="#2ca02c", label="LR-only (none)")
    axes[0].set_xlabel(r"aligned magnitude $\log_2(\mathrm{ratio})$"); axes[0].set_ylabel("collapse severity S")
    axes[0].set_ylim(0, 1); axes[0].set_title("Dose–response: severity"); axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    axes[1].plot(xm, [dr["aligned_none"]["residual_by_mag"][m] for m in mags], "o-", color="#d62728", label="aligned (none)")
    axes[1].plot(xm, [dr["scale_only_none"]["residual_by_mag"][m] for m in mags], "^:", color="#1f77b4", label="scale-only (none)")
    axes[1].set_xlabel(r"aligned magnitude $\log_2(\mathrm{ratio})$"); axes[1].set_ylabel(r"residual $|\log\rho|$")
    axes[1].set_title("Dose–response: compensation failure"); axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)
    fig.suptitle("E4-EXTREME dose–response (40 epochs, 5 seeds)")
    fig.tight_layout(); fig.savefig(figdir / f"e4_extreme_dose_response{sfx}.png", dpi=150); plt.close(fig)

    # 2) long-horizon trajectories
    def mean_traj(cls, rs, rl, norm, h, key):
        ids = [rid for rid, t in trajs.items()
               if t["cls"] == cls and np.isclose(t["r_s"], rs) and np.isclose(t["r_l"], rl)
               and t["norm"] == norm and t["epochs"] == h]
        if not ids:
            return None, None
        L = min(len(trajs[i]["epoch"]) for i in ids)
        eps = trajs[ids[0]]["epoch"][:L]
        arr = np.mean([trajs[i][key][:L] for i in ids], 0)
        return eps, arr
    H = max([40] + [h for h in df.epochs.unique() if h > 40] or [40])
    picks = [("balanced (1,1)", "balanced", 1.0, 1.0, "none", H, "#000"),
             ("scale 400×", "scale_only", 400.0, 1.0, "none", H, "#1f77b4"),
             ("aligned 20×", "aligned", 20.0, 20.0, "none", H, "#ff7f0e"),
             ("aligned 100×", "aligned", 100.0, 100.0, "none", H, "#8c564b"),
             ("aligned 400×", "aligned", 400.0, 400.0, "none", H, "#d62728"),
             ("opposed 400×", "opposed", 400.0, 1.0 / 400, "none", H, "#17becf"),
             ("aligned 400×+z", "aligned", 400.0, 400.0, "zscore", H, "#9467bd")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    base = analysis["baseline_weak"]
    for label, cls, rs, rl, norm, h, c in picks:
        eps, w = mean_traj(cls, rs, rl, norm, h, "weak")
        if eps is None:
            eps, w = mean_traj(cls, rs, rl, norm, 40, "weak")
        if eps is None:
            continue
        ls = "--" if norm == "zscore" else "-"
        axes[0].plot(eps, w, ls, color=c, label=label, lw=1.7)
        eps2, lr = mean_traj(cls, rs, rl, norm, h, "logrho")
        if eps2 is None:
            eps2, lr = mean_traj(cls, rs, rl, norm, 40, "logrho")
        axes[1].plot(eps2, lr, ls, color=c, label=label, lw=1.7)
    axes[0].axhline(0.5 * base, color="0.5", ls=":", label="collapse threshold")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("weak-modality contribution"); axes[0].grid(alpha=0.25)
    axes[0].set_title("Weak-modality contribution (long horizon)"); axes[0].legend(fontsize=7, loc="lower right")
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel(r"$\log\rho$"); axes[1].grid(alpha=0.25)
    axes[1].set_title("Activation-norm ratio (long horizon)"); axes[1].legend(fontsize=7)
    fig.suptitle("E4-EXTREME long-horizon trajectories (mean of seeds)")
    fig.tight_layout(); fig.savefig(figdir / f"e4_extreme_long_horizon_trajectories{sfx}.png", dpi=150); plt.close(fig)

    # 3) normalization decomposition
    nd = analysis["normalization_decomposition"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cats = ["scale_only", "lr_only", "aligned", "opposed"]
    none_w = [nd[c]["none_weak"] for c in cats]; z_w = [nd[c]["zscore_weak"] for c in cats]
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w / 2, none_w, w, label="none", color="#d62728")
    ax.bar(x + w / 2, z_w, w, label="z-score", color="#1f77b4")
    ax.axhline(analysis["baseline_weak"], color="k", ls="--", lw=1, label="balanced baseline")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n({max(mags):g}×)" for c in cats], fontsize=8)
    ax.set_ylabel("weak-modality contribution"); ax.set_title(f"z-score removes scale, not LR (extreme {max(mags):g}×)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(figdir / f"e4_extreme_normalization_decomposition{sfx}.png", dpi=150); plt.close(fig)

    # trajectories CSV (curated)
    tr_rows = []
    for rid, t in trajs.items():
        if t["epochs"] >= 100 or (t["cls"] in ("aligned", "scale_only", "opposed") and t["epochs"] == 40):
            for i, e in enumerate(t["epoch"]):
                tr_rows.append({"cls": t["cls"], "r_s": t["r_s"], "r_l": t["r_l"], "norm": t["norm"],
                                "seed": t["seed"], "epochs": t["epochs"], "epoch": e,
                                "weak": t["weak"][i], "logrho": t["logrho"][i], "gain": t["gain"][i]})
    pd.DataFrame(tr_rows).to_csv(ROOT / "results" / "processed" / f"e4_extreme_trajectories{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
