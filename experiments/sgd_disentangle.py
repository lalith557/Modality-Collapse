"""SGD forward-scale vs gradient-scale disentangling. See PREREGISTRATION_SGD_DISENTANGLE.md.

Repeats the validated Adam disentangling experiment under SGD (momentum 0.9, lr 1e-3 = Adam base_lr).
REUSES the validated ForwardScale/GradScale/arm_apply/arm_forward from experiments.disentangle_scale_grad
(NOT redesigned). Adds PARAMETER-UPDATE-norm measurement (||dtheta_A||/||dtheta_V||), the key cross-
optimizer signal. Adam runs are NOT rerun; the Adam CSV is loaded for the comparison table.

Usage:
    python experiments/sgd_disentangle.py --sanity-only
    python experiments/sgd_disentangle.py
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

from data.synthetic import generate_dataset
from experiments.disentangle_scale_grad import (ALPHAS, ARMS, _ci, arm_contrib, arm_forward,
                                               arm_rho, sanity_check)
from experiments.train import _optimizer, _to_device
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed


def train_one_sgd(cfg, ds, device, seed, alpha, arm):
    set_seed(seed, True)
    nc = ds.meta["n_classes"]
    data = _to_device(ds, device)
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]; a_te, v_te, y_te = data["test"]
    model = __import__("models.multimodal_model", fromlist=["build_model"]).build_model(
        cfg, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    cfg_sgd = json.loads(json.dumps(cfg)); cfg_sgd["train"]["optimizer"] = "sgd"   # <-- only change vs Adam
    opt = _optimizer(model, cfg_sgd, device)
    lf = nn.CrossEntropyLoss()
    epochs = int(cfg["train"]["epochs"]); bs = int(cfg["train"]["batch_size"]); n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    hist = []; diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device)
        ga = gb = ua = ub = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits, *_ = arm_forward(model, a_tr[idx], v_tr[idx], alpha, arm)
            loss = lf(logits, y_tr[idx])
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in ea) ** 0.5
            gb += sum(float(p.grad.pow(2).sum()) for p in ev) ** 0.5
            prev_a = [p.detach().clone() for p in ea]; prev_v = [p.detach().clone() for p in ev]
            opt.step()
            ua += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ea, prev_a)) ** 0.5
            ub += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ev, prev_v)) ** 0.5
            nb += 1
        if diverged:
            break
        re, rf = arm_rho(model, a_va, v_va, alpha, arm)
        c = arm_contrib(model, a_va, v_va, y_va, alpha, arm, nc)
        hist.append({"epoch": ep, "rho_enc": re, "rho_fusion": rf,
                     "val_dominance": c["dominance"], "val_gain": c["multimodal_gain"],
                     "val_acc_av": c["acc_av"],
                     "grad_ratio": (ga / gb) if gb > 0 else float("nan"),
                     "update_ratio": (ua / ub) if ub > 0 else float("nan"),
                     "grad_a": ga / nb, "grad_v": gb / nb, "upd_a": ua / nb, "upd_v": ub / nb})
    if diverged:
        return {"seed": seed, "alpha": alpha, "arm": arm, "diverged": True}, hist
    te = arm_contrib(model, a_te, v_te, y_te, alpha, arm, nc)
    re, rf = arm_rho(model, a_te, v_te, alpha, arm)
    comp = (1.0 - math.log(rf) / math.log(alpha)) if (alpha != 1.0 and rf > 0) else float("nan")
    final = {"seed": seed, "alpha": alpha, "arm": arm, "diverged": False,
             "acc_av": te["acc_av"], "acc_a": te["acc_a"], "acc_v": te["acc_v"],
             "delta_a": te["delta_a"], "delta_v": te["delta_v"], "dominance": te["dominance"],
             "abs_imbalance": abs(te["delta_a"] - te["delta_v"]), "multimodal_gain": te["multimodal_gain"],
             "rho_enc": re, "rho_fusion": rf, "compensation": comp,
             "grad_ratio_final": hist[-1]["grad_ratio"], "update_ratio_final": hist[-1]["update_ratio"]}
    return final, hist


def _paired(fin, alpha, arm1, arm2, col="dominance"):
    a = fin[(fin.alpha == alpha) & (fin.arm == arm1) & (~fin.diverged)].sort_values("seed")[col].values
    b = fin[(fin.alpha == alpha) & (fin.arm == arm2) & (~fin.diverged)].sort_values("seed")[col].values
    d = b - a; sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    t = sps.ttest_rel(b, a) if len(d) > 1 and sd > 0 else None
    return {"mean_diff": float(np.mean(d)), "sd": sd, "ci95": (1.96 * sd / math.sqrt(len(d))) if len(d) > 1 else 0.0,
            "cohen_d": (float(np.mean(d)) / sd) if sd > 0 else float("nan"),
            "p": float(t.pvalue) if t else float("nan"),
            "arm1_mean": float(np.mean(a)) if len(a) else float("nan"),
            "arm2_mean": float(np.mean(b)) if len(b) else float("nan")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()
    device = get_device({"type": "cuda", "allow_cpu": False})
    print("Device:", device_report(device))
    out = ROOT / "results" / "sgd_disentangle"; out.mkdir(parents=True, exist_ok=True)

    san = sanity_check(device)
    json.dump(san, open(out / "sgd_disentangle_sanity.json", "w"), indent=2, default=float)
    print("=== AUTOGRAD SANITY (optimizer-independent) ===")
    for arm in ARMS:
        s = san["scalar"][arm]
        print(f"  {arm:9s} fwd={s['forward_mult']:.4g} bwd={s['backward_mult']:.4g} pass={s['pass']}")
    print("  grad-only forward==baseline:", san["checks"]["gradient_only_forward_identical_to_baseline"],
          "| SANITY PASS:", san["pass"])
    if not san["pass"]:
        print("SANITY GATE FAILED — stopping."); return 1
    if args.sanity_only:
        print("sanity-only: gate passed, stopping before main run."); return 0

    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    seeds = cfg["seed_policy"]["seeds"]
    finals, trajs = [], []; t0 = time.time()
    for alpha in ALPHAS:
        for arm in ARMS:
            for seed in seeds:
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
                ds = generate_dataset(p)
                final, hist = train_one_sgd(cfg, ds, device, seed, alpha, arm)
                finals.append(final)
                for h in hist:
                    trajs.append({"alpha": alpha, "arm": arm, "seed": seed, **h})
            g = [f for f in finals if f["alpha"] == alpha and f["arm"] == arm and not f.get("diverged")]
            dom = np.mean([f["dominance"] for f in g]) if g else float("nan")
            nd = sum(1 for f in finals if f["alpha"] == alpha and f["arm"] == arm and f.get("diverged"))
            print(f"  alpha={alpha:g} {arm:9s} dom={dom:+.3f} diverged={nd}  ({time.time()-t0:.0f}s)")

    fin = pd.DataFrame(finals); tj = pd.DataFrame(trajs)
    fin.to_csv(out / "sgd_disentangle_primary.csv", index=False)
    tj.to_csv(out / "sgd_disentangle_trajectories.csv", index=False)
    json.dump({"optimizer": "sgd", "sgd": {"lr": cfg["train"]["lr"], "momentum": 0.9, "weight_decay": 0.0,
               "scheduler": None, "grad_clip": None}, "adam_reference": {"lr": 0.001, "betas": [0.9, 0.999],
               "eps": 1e-8, "weight_decay": 0.0}, "git_commit": git_hash(), "seeds": seeds, "alphas": ALPHAS,
               "arms": ARMS, "batch_size": cfg["train"]["batch_size"], "epochs": cfg["train"]["epochs"],
               "model": cfg["model"], "data": cfg["data"], "device": device_report(device)},
              open(out / "sgd_disentangle_config.json", "w"), indent=2, default=str)
    analyze(fin, tj, out)
    return 0


def analyze(fin, tj, out):
    ok = fin[~fin.diverged]
    rows = []
    for alpha in ALPHAS:
        for arm in ARMS:
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            for col in ["dominance", "multimodal_gain", "acc_av", "rho_enc", "rho_fusion",
                        "grad_ratio_final", "update_ratio_final"]:
                m, sd, ci = _ci(g[col].values)
                rows.append({"alpha": alpha, "arm": arm, "metric": col, "mean": m, "sd": sd, "ci95": ci})
    pd.DataFrame(rows).to_csv(out / "sgd_disentangle_agg.csv", index=False)

    a = {"n_diverged": int(fin.diverged.sum()), "by_alpha": {}}
    for alpha in [10.0, 100.0]:
        base = float(ok[(ok.alpha == alpha) & (ok.arm == "baseline")].dominance.mean())
        block = {"baseline_dominance": base}
        for arm in ARMS:
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            block[arm] = {"dominance": m, "sd": sd, "ci95": ci, "shift_vs_baseline": m - base,
                          "gain": float(g.multimodal_gain.mean()), "acc_av": float(g.acc_av.mean()),
                          "rho_enc": float(g.rho_enc.mean()), "rho_fusion": float(g.rho_fusion.mean()),
                          "grad_ratio": float(g.grad_ratio_final.mean()),
                          "update_ratio": float(g.update_ratio_final.mean())}
        block["cmp"] = {"forward_vs_baseline": _paired(fin, alpha, "baseline", "forward"),
                        "gradient_vs_baseline": _paired(fin, alpha, "baseline", "gradient"),
                        "ordinary_vs_baseline": _paired(fin, alpha, "baseline", "ordinary"),
                        "ordinary_vs_forward": _paired(fin, alpha, "forward", "ordinary"),
                        "ordinary_vs_gradient": _paired(fin, alpha, "gradient", "ordinary")}
        a["by_alpha"][alpha] = block

    b = a["by_alpha"][100.0]; thr = 0.02
    fs = abs(b["cmp"]["forward_vs_baseline"]["mean_diff"]); gs = abs(b["cmp"]["gradient_vs_baseline"]["mean_diff"])
    os_ = abs(b["cmp"]["ordinary_vs_baseline"]["mean_diff"])
    if fs < thr and gs < thr and os_ < thr:
        case = "CASE_null — no arm shifts contribution (SGD config under-learns or fully compensates)"
    elif gs > thr and fs > thr:
        case = "CASE_C — mixed (both forward and gradient pathways shift contribution under SGD)"
    elif gs > thr and fs < thr:
        case = "CASE_A/gradient — gradient-only shifts contribution under SGD (effective-LR pathway)"
    elif fs > thr and gs < thr:
        case = "CASE_E — forward shifts, isolated gradient scaling insufficient (as under Adam)"
    else:
        case = "CASE_mixed/partial"
    a["classification"] = case
    a["seed_sd_alpha100"] = {arm: float(ok[(ok.alpha == 100.0) & (ok.arm == arm)].dominance.std(ddof=1)) for arm in ARMS}

    # Adam-vs-SGD comparison (load Adam results; do NOT rerun Adam)
    adam_path = ROOT / "results" / "disentangle" / "disentangle_primary.csv"
    cmp_rows = []
    if adam_path.exists():
        adam = pd.read_csv(adam_path); adam = adam[~adam.diverged]
        for alpha in [10.0, 100.0]:
            abase = float(adam[(adam.alpha == alpha) & (adam.arm == "baseline")].dominance.mean())
            sbase = float(ok[(ok.alpha == alpha) & (ok.arm == "baseline")].dominance.mean())
            for arm in ARMS:
                ag = adam[(adam.alpha == alpha) & (adam.arm == arm)]
                sg = ok[(ok.alpha == alpha) & (ok.arm == arm)]
                cmp_rows.append({"alpha": alpha, "arm": arm,
                                 "adam_dom_shift": float(ag.dominance.mean()) - abase,
                                 "sgd_dom_shift": float(sg.dominance.mean()) - sbase,
                                 "adam_grad_ratio": float(ag.grad_ratio_final.mean()),
                                 "sgd_grad_ratio": float(sg.grad_ratio_final.mean()),
                                 "sgd_update_ratio": float(sg.update_ratio_final.mean()),
                                 "adam_gain": float(ag.multimodal_gain.mean()), "sgd_gain": float(sg.multimodal_gain.mean()),
                                 "adam_rho_enc": float(ag.rho_enc.mean()), "sgd_rho_enc": float(sg.rho_enc.mean())})
        pd.DataFrame(cmp_rows).to_csv(out / "sgd_disentangle_adam_vs_sgd.csv", index=False)
        a["adam_available"] = True
    else:
        a["adam_available"] = False
    json.dump(a, open(out / "sgd_disentangle_analysis.json", "w"), indent=2, default=float)

    print("\n=== SGD dominance by arm (shift vs baseline) ===")
    for alpha in [10.0, 100.0]:
        print(f" alpha={alpha:g}: baseline dom={a['by_alpha'][alpha]['baseline_dominance']:+.3f}")
        for arm in ARMS:
            x = a["by_alpha"][alpha][arm]
            print(f"   {arm:9s} dom={x['dominance']:+.3f} shift={x['shift_vs_baseline']:+.3f} gain={x['gain']:+.3f} "
                  f"rho_enc={x['rho_enc']:.2f} grad_ratio={x['grad_ratio']:.2f} update_ratio={x['update_ratio']:.2f}")
    print("CLASSIFICATION:", a["classification"])
    if cmp_rows:
        print("\n=== Adam vs SGD (dominance shift | grad ratio | sgd update ratio), alpha=100 ===")
        for r in [r for r in cmp_rows if r["alpha"] == 100.0]:
            print(f"   {r['arm']:9s} adam_shift={r['adam_dom_shift']:+.3f} sgd_shift={r['sgd_dom_shift']:+.3f} "
                  f"| adam_gr={r['adam_grad_ratio']:.1f} sgd_gr={r['sgd_grad_ratio']:.1f} sgd_upd={r['sgd_update_ratio']:.2f}")
    figures(fin, tj, a, cmp_rows, out)


def figures(fin, tj, a, cmp_rows, out):
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    ok = fin[~fin.diverged]
    COL = {"baseline": "#000000", "forward": "#ff7f0e", "gradient": "#1f77b4", "ordinary": "#d62728"}

    # Fig 1: dominance shift by arm (alpha 10 & 100) with per-seed points
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, alpha in zip(axes, [10.0, 100.0]):
        base = a["by_alpha"][alpha]["baseline_dominance"]
        for j, arm in enumerate(ARMS):
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            ax.bar(j, m - base, yerr=ci, capsize=3, color=COL[arm], alpha=0.8)
            ax.scatter([j] * len(g), g.dominance.values - base, color="0.3", s=14, zorder=3)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_title(f"SGD alpha={alpha:g}: dominance shift vs baseline"); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("SGD forward vs gradient pathway: dominance shift (A scaled, 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "sgd_dominance.png", dpi=150); plt.close(fig)

    # Fig 2: grad ratio & update ratio by arm (alpha=100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, tit in zip(axes, ["grad_ratio_final", "update_ratio_final"], ["encoder gradient ratio", "parameter-update ratio"]):
        for j, arm in enumerate(ARMS):
            g = ok[(ok.alpha == 100.0) & (ok.arm == arm)]
            m, sd, ci = _ci(g[key].values)
            ax.bar(j, m, yerr=ci, capsize=3, color=COL[arm]); ax.scatter([j] * len(g), g[key].values, color="0.3", s=12, zorder=3)
        ax.axhline(1, color="0.6", ls=":"); ax.set_yscale("log"); ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_title(f"SGD alpha=100: {tit}  ||A||/||V||"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "sgd_grad_update_ratio.png", dpi=150); plt.close(fig)

    # Fig 3: Adam vs SGD dominance shift (alpha=100)
    if cmp_rows:
        r100 = {r["arm"]: r for r in cmp_rows if r["alpha"] == 100.0}
        fig, ax = plt.subplots(figsize=(8, 4.6)); x = np.arange(len(ARMS)); w = 0.38
        ax.bar(x - w / 2, [r100[a_]["adam_dom_shift"] for a_ in ARMS], w, label="Adam", color="#9467bd")
        ax.bar(x + w / 2, [r100[a_]["sgd_dom_shift"] for a_ in ARMS], w, label="SGD", color="#8c564b")
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(ARMS)
        ax.set_ylabel("dominance shift vs baseline"); ax.set_title("Adam vs SGD: dominance shift by arm (alpha=100)")
        ax.legend(); ax.grid(alpha=0.25, axis="y")
        fig.tight_layout(); fig.savefig(fig_dir / "sgd_adam_vs_sgd.png", dpi=150); plt.close(fig)

    # Fig 4: trajectories alpha=100 (dominance, gain, grad_ratio, update_ratio)
    tn = tj[tj.alpha == 100.0]
    fig, axx = plt.subplots(1, 4, figsize=(18, 4.2))
    for k, key in enumerate(["val_dominance", "val_gain", "grad_ratio", "update_ratio"]):
        for arm in ARMS:
            g = tn[tn.arm == arm]
            if len(g):
                m = g.groupby("epoch")[key].mean(); axx[k].plot(m.index, m.values, color=COL[arm], label=arm, lw=1.7)
        axx[k].set_title(key); axx[k].set_xlabel("epoch"); axx[k].grid(alpha=0.25); axx[k].legend(fontsize=7)
        if key in ("grad_ratio", "update_ratio"):
            axx[k].set_yscale("log"); axx[k].axhline(1, color="0.6", ls=":")
    fig.suptitle("SGD alpha=100 training trajectories by arm (mean of 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "sgd_trajectories.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
