"""E5 - OGM-GE mechanistic comparison. See PREREGISTRATION_E5.md.

Four arms {vanilla, z-score, OGM-GE, z-score+OGM-GE} across targeted conditions
(scale-only / LR-only / aligned / opposed), testing whether normalization (scale
channel) and gradient modulation (optimization channel) correct separable
components of modality imbalance. Reuses the E4 generator/architecture/frozen
collapse definition. Records gradient norms + OGM coefficients.

Usage:
    python experiments/e5_ogm.py --tag smoke --seeds 2 --short
    python experiments/e5_ogm.py --tag primary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sps

from analysis.compensation import compensation_from_history
from analysis.metrics import contribution_metrics
from baselines.ogm_ge import apply_modulation, ogm_coefficients, per_modality_scores
from data.synthetic import generate_dataset
from experiments.train import _optimizer, _to_device
from interventions.diagnostic import measure_rho
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import load_config
from msc.seed import set_seed

BASE_LR = 0.001
FINAL_WINDOW = 5
ALPHA, GAMMA = 0.1, 0.5


def train_e5(cfg, ds, device, seed, arm, epochs):
    set_seed(seed, deterministic=True)
    norm = "zscore" if arm in ("zscore", "both") else "none"
    ogm_on = arm in ("ogm", "both")
    cfg = json.loads(json.dumps(cfg))
    cfg["model"]["normalization"] = norm
    cfg["train"]["epochs"] = epochs
    nc = ds.meta["n_classes"]
    data = _to_device(ds, device)
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]; a_te, v_te, y_te = data["test"]

    model = build_model(cfg, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    if norm == "zscore":
        with torch.no_grad():
            h_a, h_b = model.encode(a_tr, v_tr)
            if hasattr(model.norm_a, "fit"):
                model.norm_a.fit(h_a); model.norm_v.fit(h_b)
    opt = _optimizer(model, cfg, device)
    loss_fn = nn.CrossEntropyLoss()
    enc_a_p = list(model.enc_a.parameters()); enc_b_p = list(model.enc_v.parameters())
    n = y_tr.numel(); bs = int(cfg["train"]["batch_size"])
    history = []

    def epoch_row(ep, gnorm_a, gnorm_b, k_a, k_b):
        row = dict(measure_rho(model, a_va, v_va))
        c = contribution_metrics(model, a_va, v_va, y_va, nc)
        row.update(epoch=ep, val_delta_a=c["delta_a"], val_delta_v=c["delta_v"],
                   val_gain=c["multimodal_gain"], val_acc_av=c["acc_av"], val_f1_av=c["f1_av"],
                   gnorm_a=gnorm_a, gnorm_b=gnorm_b,
                   grad_ratio=(gnorm_a / gnorm_b if gnorm_b > 0 else float("nan")),
                   k_a=k_a, k_b=k_b)
        return row

    history.append(epoch_row(0, float("nan"), float("nan"), 1.0, 1.0))
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ga, gb, kas, kbs, nb = 0.0, 0.0, 0.0, 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits, (h_a, h_b) = model(a_tr[idx], v_tr[idx], return_reps=True)
            k_a, k_b = 1.0, 1.0
            if ogm_on:
                s_a, s_b = per_modality_scores(model, h_a.detach(), h_b.detach())
                k_a, k_b, _ = ogm_coefficients(s_a, s_b, y_tr[idx], ALPHA)
            loss = loss_fn(logits, y_tr[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            pre_a, pre_b = apply_modulation(enc_a_p, enc_b_p, k_a if ogm_on else 1.0,
                                            k_b if ogm_on else 1.0, GAMMA if ogm_on else 0.0)
            opt.step()
            ga += pre_a; gb += pre_b; kas += k_a; kbs += k_b; nb += 1
        history.append(epoch_row(epoch, ga / nb, gb / nb, kas / nb, kbs / nb))

    final = contribution_metrics(model, a_te, v_te, y_te, nc)
    win = list(range(max(1, epochs - FINAL_WINDOW + 1), epochs + 1))
    win_min = {int(r["epoch"]): min(r["val_delta_a"], r["val_delta_v"]) for r in history if int(r["epoch"]) in win}
    weak_window = float(np.mean(list(win_min.values())))
    comp = compensation_from_history(history, math.log(ds.meta["alpha_ratio"]), epochs)
    return {"final": final, "weak_window": weak_window, "win_min": win_min,
            "residual_abs_logrho": comp["residual_abs_logrho"], "final_rho": history[-1]["rho"],
            "rho_early": next((r["rho"] for r in history if r["epoch"] == 5), float("nan")),
            "history": history}


def conditions(short=False):
    """Each entry: (label, scale_ratio, lr_ratio, extra_data_params).
    The 'difficulty' condition is a POSITIVE CONTROL for OGM fidelity: equal LR and
    scale, but modality A is intrinsically easier (larger prototype margin) so the
    fast modality dominates via loss dynamics -- OGM's NATIVE target setting."""
    M = 100
    if short:
        return [("balanced", 1.0, 1.0, {}), ("aligned_strong", M, M, {}), ("lr_only", 1.0, M, {}),
                ("difficulty", 1.0, 1.0, {"margin_a": 2.5, "margin_v": 1.0})]
    return [
        ("balanced", 1.0, 1.0, {}),
        ("scale_only", 100.0, 1.0, {}), ("scale_only", 1 / 100.0, 1.0, {}),
        ("lr_only", 1.0, 100.0, {}), ("lr_only", 1.0, 1 / 100.0, {}),
        ("aligned_mod", 20.0, 20.0, {}), ("aligned_mod", 1 / 20.0, 1 / 20.0, {}),
        ("aligned_strong", 100.0, 100.0, {}), ("aligned_strong", 1 / 100.0, 1 / 100.0, {}),
        ("opposed", 100.0, 1 / 100.0, {}), ("opposed", 1 / 100.0, 100.0, {}),
        # positive control for OGM fidelity (info asymmetry, equal LR & scale):
        ("difficulty", 1.0, 1.0, {"margin_a": 2.5, "margin_v": 1.0}),
        ("difficulty", 1.0, 1.0, {"margin_a": 1.0, "margin_v": 2.5}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--short", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    device = get_device(cfg["device"])
    print("Device:", device_report(device))
    seeds = list(range(args.seeds))
    arms = ["vanilla", "zscore", "ogm", "both"]
    conds = conditions(args.short)
    epochs = 40

    rows, trajs = [], {}
    rid = 0
    t0 = time.time()
    for cls, rs, rl, extra in conds:
        for arm in arms:
            for seed in seeds:
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=rs ** 0.5, alpha_v=rs ** -0.5)
                p.update(extra)      # e.g. margin_a/margin_v for the difficulty control
                ds = generate_dataset(p)
                cfg2 = json.loads(json.dumps(cfg))
                cfg2["train"]["lr_a"] = BASE_LR * (rl ** 0.5); cfg2["train"]["lr_v"] = BASE_LR * (rl ** -0.5)
                res = train_e5(cfg2, ds, device, seed, arm, epochs)
                f = res["final"]
                row = {"run_id": rid, "condition": cls, "direction": "A" if math.log2(rs if rs != 1 else rl) >= 0 else "V",
                       "margin_a": ds.meta["margin_a"], "margin_v": ds.meta["margin_v"],
                       "scale_ratio": rs, "lr_ratio": rl, "method": arm, "seed": seed, "epoch": epochs,
                       "rho": res["final_rho"], "rho_early": res["rho_early"],
                       "delta_a": f["delta_a"], "delta_v": f["delta_v"],
                       "weak_contribution": res["weak_window"], "absdom": abs(f["delta_a"] - f["delta_v"]),
                       "multimodal_gain": f["multimodal_gain"], "accuracy": f["acc_av"], "macro_f1": f["f1_av"],
                       "compensation_residual": res["residual_abs_logrho"],
                       "grad_ratio": res["history"][-1]["grad_ratio"],
                       "k_a_final": res["history"][-1]["k_a"], "k_b_final": res["history"][-1]["k_b"]}
                rows.append(row)
                trajs[rid] = {"condition": cls, "scale_ratio": rs, "lr_ratio": rl, "method": arm, "seed": seed,
                              "win_min": res["win_min"],
                              **{k: [r[k] for r in res["history"]] for k in
                                 ("epoch", "rho", "val_delta_a", "val_delta_v", "val_gain", "grad_ratio", "k_a", "k_b")}}
                rid += 1
        print(f"  done {cls:14s} rs={rs:g} rl={rl:g}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    E5 = ROOT / "results" / "e5"; E5.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.tag == "primary" else f"_{args.tag}"
    df.to_csv(E5 / f"e5_primary{sfx}.csv", index=False)
    json.dump({"alpha": ALPHA, "gamma": GAMMA, "epochs": epochs, "seeds": seeds, "arms": arms,
               "conditions": conds, "base_lr": BASE_LR, "fusion": cfg["model"]["fusion"],
               "collapse": {"final_window": FINAL_WINDOW, "halved": 0.5, "stable": 0.6}},
              open(E5 / f"e5_config{sfx}.json", "w"), indent=2, default=str)

    baseline = float(df[(df.condition == "balanced") & (df.method == "vanilla")].weak_contribution.mean())
    df["severity"] = np.clip(1 - df.weak_contribution / baseline, 0, 1)

    # aggregated
    agg = df.groupby(["condition", "scale_ratio", "lr_ratio", "method"]).agg(
        weak_mean=("weak_contribution", "mean"), weak_sd=("weak_contribution", "std"),
        severity_mean=("severity", "mean"), gain_mean=("multimodal_gain", "mean"),
        acc_mean=("accuracy", "mean"), f1_mean=("macro_f1", "mean"),
        residual_mean=("compensation_residual", "mean"), grad_ratio_mean=("grad_ratio", "mean"), n=("seed", "count")).reset_index()
    agg["weak_ci95"] = 1.96 * agg["weak_sd"] / np.sqrt(agg["n"])
    agg.to_csv(E5 / f"e5_aggregated{sfx}.csv", index=False)

    analysis = analyze(df, baseline)
    json.dump(analysis, open(E5 / f"e5_analysis{sfx}.json", "w"), indent=2, default=float)
    make_figures(df, trajs, analysis, baseline, ROOT / "results" / "figures", sfx)

    print("\n===== E5 =====  baseline_weak:", round(baseline, 3))
    print(analysis["main_table_weak"])
    print("classification:", analysis["classification"])


def _paired(df, cond_sr_lr, m1, m2, col="weak_contribution"):
    (cls, rs, rl) = cond_sr_lr
    a = df[(df.condition == cls) & np.isclose(df.scale_ratio, rs) & np.isclose(df.lr_ratio, rl) & (df.method == m1)].sort_values("seed")
    b = df[(df.condition == cls) & np.isclose(df.scale_ratio, rs) & np.isclose(df.lr_ratio, rl) & (df.method == m2)].sort_values("seed")
    da = b[col].values - a[col].values
    m = float(np.mean(da)); sd = float(np.std(da, ddof=1)) if len(da) > 1 else 0.0
    d = m / sd if sd > 0 else float("nan")
    t = sps.ttest_rel(b[col].values, a[col].values) if len(da) > 1 else None
    return {"mean_diff": m, "ci95": 1.96 * sd / math.sqrt(len(da)) if len(da) else float("nan"),
            "cohen_d": d, "p": float(t.pvalue) if t else float("nan"),
            "m1_mean": float(a[col].mean()), "m2_mean": float(b[col].mean())}


def analyze(df, baseline):
    order = [("balanced", 1.0, 1.0), ("scale_only", 100.0, 1.0), ("lr_only", 1.0, 100.0),
             ("aligned_mod", 20.0, 20.0), ("aligned_strong", 100.0, 100.0), ("opposed", 100.0, 1 / 100.0)]
    labels = {"balanced": "Balanced", "scale_only": "Scale-only 100x", "lr_only": "LR-only 100x",
              "aligned_mod": "Aligned 20x", "aligned_strong": "Aligned 100x", "opposed": "Opposed 100x"}
    tbl_weak, tbl_acc = {}, {}
    for cls, rs, rl in order:
        row_w, row_a = {}, {}
        for m in ["vanilla", "zscore", "ogm", "both"]:
            g = df[(df.condition == cls) & np.isclose(df.scale_ratio, rs) & np.isclose(df.lr_ratio, rl) & (df.method == m)]
            row_w[m] = round(float(g.weak_contribution.mean()), 3)
            row_a[m] = round(float(g.accuracy.mean()), 3)
        tbl_weak[labels[cls]] = row_w; tbl_acc[labels[cls]] = row_a

    comparisons = {
        "C1_scaleonly_vanilla_vs_zscore": _paired(df, ("scale_only", 100.0, 1.0), "vanilla", "zscore"),
        "C1_scaleonly_vanilla_vs_ogm": _paired(df, ("scale_only", 100.0, 1.0), "vanilla", "ogm"),
        "C2_lronly_vanilla_vs_ogm": _paired(df, ("lr_only", 1.0, 100.0), "vanilla", "ogm"),
        "C2_lronly_vanilla_vs_zscore": _paired(df, ("lr_only", 1.0, 100.0), "vanilla", "zscore"),
        "C4_aligned100_vanilla_vs_ogm": _paired(df, ("aligned_strong", 100.0, 100.0), "vanilla", "ogm"),
        "C4_aligned100_zscore_vs_ogm": _paired(df, ("aligned_strong", 100.0, 100.0), "zscore", "ogm"),
        "C4_aligned100_ogm_vs_both": _paired(df, ("aligned_strong", 100.0, 100.0), "ogm", "both"),
        "C4_aligned100_vanilla_vs_both": _paired(df, ("aligned_strong", 100.0, 100.0), "vanilla", "both"),
    }

    # positive control (difficulty asymmetry, equal LR & scale) — OGM's native setting
    pos = {}
    dc = df[df.condition == "difficulty"]
    if len(dc):
        for lab, ma in [("A_easier", 2.5), ("V_easier", 1.0)]:
            sub = dc[np.isclose(dc.margin_a, ma)]
            if len(sub):
                pos[lab] = {m: round(float(sub[sub.method == m].weak_contribution.mean()), 3)
                            for m in ["vanilla", "zscore", "ogm", "both"]}
                v = sub[sub.method == "vanilla"].sort_values("seed").weak_contribution.values
                o = sub[sub.method == "ogm"].sort_values("seed").weak_contribution.values
                pos[lab]["ogm_minus_vanilla"] = round(float(np.mean(o - v)), 3)

    # grad ratio at aligned 100x (mechanism)
    def gratio(m):
        g = df[(df.condition == "aligned_strong") & np.isclose(df.scale_ratio, 100.0) & (df.method == m)]
        return float(g.grad_ratio.mean())
    grad_mech = {m: gratio(m) for m in ["vanilla", "zscore", "ogm", "both"]}

    # classification
    scale_zscore = comparisons["C1_scaleonly_vanilla_vs_zscore"]["mean_diff"]
    scale_ogm = comparisons["C1_scaleonly_vanilla_vs_ogm"]["mean_diff"]
    lr_ogm = comparisons["C2_lronly_vanilla_vs_ogm"]["mean_diff"]
    lr_zscore = comparisons["C2_lronly_vanilla_vs_zscore"]["mean_diff"]
    al_ogm = comparisons["C4_aligned100_vanilla_vs_ogm"]["mean_diff"]
    al_both = comparisons["C4_aligned100_vanilla_vs_both"]["mean_diff"]
    thr = 0.3 * baseline
    zscore_fixes_scale = scale_zscore > thr
    ogm_fixes_lr = lr_ogm > thr
    zscore_fails_lr = lr_zscore < thr
    combined_best = al_both > max(al_ogm, comparisons["C4_aligned100_zscore_vs_ogm"]["m1_mean"] - tbl_weak["Aligned 100x"]["vanilla"]) + 0.5 * thr
    if zscore_fixes_scale and ogm_fixes_lr and zscore_fails_lr:
        cls = "E5-A — MECHANISTIC SEPARATION SUPPORTED" if al_both > al_ogm + 0.5 * thr else "E5-B — OGM-GE fixes LR channel, z-score fixes scale channel (decomposition supported)"
    elif ogm_fixes_lr and not zscore_fixes_scale:
        cls = "E5-C — methods act on overlapping phenomenon"
    elif not ogm_fixes_lr:
        cls = "E5-D — OGM-GE does not correct LR-driven collapse"
    else:
        cls = "E5-B — partial decomposition"
    # OGM fidelity: is the modulation ACTIVE (k<1 where a modality dominates)?
    ka_aligned = df[(df.condition == "aligned_strong") & (df.method == "ogm")].k_a_final.mean()
    kb_aligned = df[(df.condition == "aligned_strong") & (df.method == "ogm")].k_b_final.mean()
    ogm_modulation_active = bool(min(ka_aligned, kb_aligned) < 0.9)
    ogm_works_native = False
    if "A_easier" in pos:
        ogm_works_native = pos["A_easier"].get("ogm_minus_vanilla", 0) > 0.02 or pos.get("V_easier", {}).get("ogm_minus_vanilla", 0) > 0.02
    # E5-E only if modulation is INACTIVE (truly uncertain implementation).
    if not ogm_modulation_active:
        cls = "E5-E — OGM implementation uncertain (modulation coefficients stayed ~1)"
    elif not ogm_fixes_lr and not ogm_works_native:
        cls = "E5-D — OGM-GE does not correct the induced collapse (modulation active but ineffective)"

    return {"baseline_weak": baseline, "main_table_weak": tbl_weak, "main_table_acc": tbl_acc,
            "comparisons": comparisons, "grad_ratio_aligned100": grad_mech,
            "positive_control": pos, "ogm_works_in_native_setting": bool(ogm_works_native),
            "ogm_modulation_active": ogm_modulation_active,
            "ogm_k_aligned": {"k_a": float(ka_aligned), "k_b": float(kb_aligned)},
            "flags": {"zscore_fixes_scale": bool(zscore_fixes_scale), "ogm_fixes_lr": bool(ogm_fixes_lr),
                      "zscore_fails_lr": bool(zscore_fails_lr)}, "classification": cls}


def make_figures(df, trajs, analysis, baseline, figdir, sfx):
    figdir.mkdir(parents=True, exist_ok=True)
    COL = {"vanilla": "#d62728", "zscore": "#1f77b4", "ogm": "#2ca02c", "both": "#9467bd"}

    def mean_ci(g, col="weak_contribution"):
        if len(g) == 0:
            return float("nan"), 0.0
        v = g[col].values
        return float(np.mean(v)), (1.96 * float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else 0.0)
    order = [("balanced", 1.0, 1.0, "Balanced"), ("scale_only", 100.0, 1.0, "Scale100x"),
             ("lr_only", 1.0, 100.0, "LR100x"), ("aligned_mod", 20.0, 20.0, "Aligned20x"),
             ("aligned_strong", 100.0, 100.0, "Aligned100x"), ("opposed", 100.0, 1 / 100.0, "Opposed100x")]

    # Fig 2 / 5: intervention comparison (weak contribution) across conditions x methods
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(order)); w = 0.2
    for j, m in enumerate(["vanilla", "zscore", "ogm", "both"]):
        means, cis = [], []
        for cls, rs, rl, _ in order:
            g = df[(df.condition == cls) & np.isclose(df.scale_ratio, rs) & np.isclose(df.lr_ratio, rl) & (df.method == m)]
            mm, cc = mean_ci(g)
            means.append(mm); cis.append(cc)
        ax.bar(x + (j - 1.5) * w, means, w, yerr=cis, capsize=2, label=m, color=COL[m])
    ax.axhline(baseline, color="k", ls="--", lw=1, label="balanced baseline")
    ax.axhline(0.5 * baseline, color="0.5", ls=":", lw=1, label="collapse threshold")
    ax.set_xticks(x); ax.set_xticklabels([o[3] for o in order], fontsize=8)
    ax.set_ylabel("weak-modality contribution"); ax.set_title("E5 intervention comparison (weak-modality contribution)")
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(figdir / f"e5_intervention_comparison{sfx}.png", dpi=150); plt.close(fig)

    # Fig 1: mechanism decomposition — severity by condition x method (scale/lr/aligned/opposed)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    conds = [("scale_only", 100.0, 1.0, "Scale-only"), ("lr_only", 1.0, 100.0, "LR-only"),
             ("aligned_strong", 100.0, 100.0, "Aligned"), ("opposed", 100.0, 1 / 100.0, "Opposed")]
    x = np.arange(len(conds))
    for j, m in enumerate(["vanilla", "zscore", "ogm", "both"]):
        sev = []
        for cls, rs, rl, _ in conds:
            g = df[(df.condition == cls) & np.isclose(df.scale_ratio, rs) & np.isclose(df.lr_ratio, rl) & (df.method == m)]
            sev.append(g.severity.mean())
        ax.bar(x + (j - 1.5) * w, sev, w, label=m, color=COL[m])
    ax.set_xticks(x); ax.set_xticklabels([c[3] for c in conds]); ax.set_ylabel("collapse severity S")
    ax.set_title("E5 mechanism decomposition (100x): which channel does each method fix?")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(figdir / f"e5_mechanism_decomposition{sfx}.png", dpi=150); plt.close(fig)

    # Fig 3: aligned 100x trajectories (weak + rho) per method
    def mean_traj(cls, rs, rl, m, key):
        ids = [i for i, t in trajs.items() if t["condition"] == cls and np.isclose(t["scale_ratio"], rs)
               and np.isclose(t["lr_ratio"], rl) and t["method"] == m]
        if not ids:
            return None, None
        L = min(len(trajs[i]["epoch"]) for i in ids)
        if key == "weak":
            arr = np.mean([[min(t["val_delta_a"][k], t["val_delta_v"][k]) for k in range(L)] for t in [trajs[i] for i in ids]], 0)
        elif key == "logrho":
            arr = np.mean([[math.log(t["rho"][k]) for k in range(L)] for t in [trajs[i] for i in ids]], 0)
        else:
            arr = np.mean([t[key][:L] for t in [trajs[i] for i in ids]], 0)
        return trajs[ids[0]]["epoch"][:L], arr
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for m in ["vanilla", "zscore", "ogm", "both"]:
        e, w = mean_traj("aligned_strong", 100.0, 100.0, m, "weak")
        if e: axes[0].plot(e, w, color=COL[m], label=m, lw=1.8)
        e2, lr = mean_traj("aligned_strong", 100.0, 100.0, m, "logrho")
        if e2: axes[1].plot(e2, lr, color=COL[m], label=m, lw=1.8)
    axes[0].axhline(0.5 * baseline, color="0.5", ls=":"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("weak contribution")
    axes[0].set_title("Aligned 100x: weak-modality contribution"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    axes[1].axhline(0, color="0.6", lw=0.8); axes[1].set_xlabel("epoch"); axes[1].set_ylabel(r"$\log\rho$")
    axes[1].set_title("Aligned 100x: activation-norm ratio"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figdir / f"e5_aligned100_trajectories{sfx}.png", dpi=150); plt.close(fig)

    # Fig 4: gradient mechanism — grad ratio & modulation coeff over epochs (aligned 100x)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m in ["vanilla", "ogm"]:
        e, gr = mean_traj("aligned_strong", 100.0, 100.0, m, "grad_ratio")
        if e: axes[0].plot(e[1:], gr[1:], color=COL[m], label=f"{m}", lw=1.8)
    axes[0].axhline(1.0, color="0.6", ls="--"); axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel(r"pre-modulation grad ratio $\|g_A\|/\|g_V\|$")
    axes[0].set_title("Aligned 100x: encoder gradient imbalance"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    for m in ["ogm", "both"]:
        e, ka = mean_traj("aligned_strong", 100.0, 100.0, m, "k_a")
        e, kb = mean_traj("aligned_strong", 100.0, 100.0, m, "k_b")
        if e:
            axes[1].plot(e[1:], ka[1:], color=COL[m], ls="-", label=f"{m} k_A", lw=1.6)
            axes[1].plot(e[1:], kb[1:], color=COL[m], ls="--", label=f"{m} k_V", lw=1.6)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("OGM modulation coefficient k")
    axes[1].set_title("Aligned 100x: OGM coefficients"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figdir / f"e5_gradient_mechanism{sfx}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
