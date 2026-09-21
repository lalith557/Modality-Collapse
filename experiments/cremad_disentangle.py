"""CREMA-D forward-scale vs gradient-scale disentangling (Phase 2). See PREREGISTRATION_CREMAD_DISENTANGLE.md.

Real-data test of the synthetic forward-vs-gradient decomposition, under the SAME Adam as H6.
Modality A = audio (scaled), B = video. REUSES validated ForwardScale/GradScale from
experiments.disentangle_scale_grad and the frozen CREMA-D cache/model. Batched eval (chunk 512).
Records encoder gradient ratio AND parameter-update ratio. All prior experiments immutable.

Usage:
    python experiments/cremad_disentangle.py --sanity-only
    python experiments/cremad_disentangle.py
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

from analysis.metrics import macro_f1
from experiments.cremad_run import _chunks, load_features
from experiments.disentangle_scale_grad import arm_apply
from models.cremad_model import CremadMultimodalModel, count_params
from models.fusion import fuse
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ARMS = ["baseline", "forward", "gradient", "ordinary"]
ALPHAS = [10.0, 100.0]


def _reps(model, x_a, x_v):
    return model.norm_a(model.enc_a(x_a)), model.norm_v(model.enc_v(x_v))


def arm_logits(model, x_a, x_v, alpha, arm, mask_a=1.0, mask_v=1.0):
    h_a, h_v = _reps(model, x_a, x_v)
    h_a = arm_apply(h_a, alpha, arm)
    return model.head(fuse(model.fusion, mask_a * h_a, mask_v * h_v))


@torch.no_grad()
def arm_contrib(model, a, v, y, alpha, arm, nc):
    was = model.training; model.eval()
    out = {}
    for name, (ma, mv) in {"both": (1.0, 1.0), "a": (1.0, 0.0), "v": (0.0, 1.0)}.items():
        preds = []
        for sl in _chunks(a.shape[0]):
            preds.append(arm_logits(model, a[sl], v[sl], alpha, arm, ma, mv).argmax(1))
        preds = torch.cat(preds)
        out[name] = (float((preds == y).float().mean()), macro_f1(preds, y, nc))
    if was:
        model.train()
    acc_av, acc_a, acc_v = out["both"][0], out["a"][0], out["v"][0]
    return {"acc_av": acc_av, "acc_a": acc_a, "acc_v": acc_v,
            "f1_av": out["both"][1], "delta_a": acc_av - acc_v, "delta_v": acc_av - acc_a,
            "multimodal_gain": acc_av - max(acc_a, acc_v), "dominance": acc_a - acc_v}


@torch.no_grad()
def arm_rho(model, a, v, alpha, arm):
    was = model.training; model.eval()
    sa = sv = sp = ntot = 0.0
    for sl in _chunks(a.shape[0]):
        h_a, h_v = _reps(model, a[sl], v[sl])
        h_post = arm_apply(h_a, alpha, arm)
        sa += float(h_a.norm(dim=1).sum()); sv += float(h_v.norm(dim=1).sum())
        sp += float(h_post.norm(dim=1).sum()); ntot += h_a.shape[0]
    if was:
        model.train()
    rho_enc = sa / sv if sv > 0 else float("nan")
    rho_fus = sp / sv if sv > 0 else float("nan")
    return rho_enc, rho_fus


def train_one(cfg, feats, device, seed, alpha, arm):
    set_seed(seed, True)
    nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]; a_va, v_va, y_va = feats["val"]; a_te, v_te, y_te = feats["test"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                                  normalization="none", head=cfg["model"]["head"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    lf = nn.CrossEntropyLoss()
    epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    hist = []; t0 = time.time(); diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); ga = gb = ua = ub = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lf(arm_logits(model, a_tr[idx], v_tr[idx], alpha, arm), y_tr[idx])   # scale IN-GRAPH
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in ea) ** 0.5
            gb += sum(float(p.grad.pow(2).sum()) for p in ev) ** 0.5
            pa = [p.detach().clone() for p in ea]; pv = [p.detach().clone() for p in ev]
            opt.step()
            ua += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ea, pa)) ** 0.5
            ub += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ev, pv)) ** 0.5
            nb += 1
        if diverged:
            break
        re, rf = arm_rho(model, a_va, v_va, alpha, arm)
        c = arm_contrib(model, a_va, v_va, y_va, alpha, arm, nc)
        hist.append({"epoch": ep, "rho_enc": re, "rho_fusion": rf, "val_dominance": c["dominance"],
                     "val_gain": c["multimodal_gain"], "val_acc_av": c["acc_av"],
                     "grad_ratio": (ga / gb) if gb > 0 else float("nan"),
                     "update_ratio": (ua / ub) if ub > 0 else float("nan")})
    if diverged:
        return {"seed": seed, "alpha": alpha, "arm": arm, "diverged": True}, hist
    te = arm_contrib(model, a_te, v_te, y_te, alpha, arm, nc)
    re, rf = arm_rho(model, a_te, v_te, alpha, arm)
    comp = (1.0 - math.log(rf) / math.log(alpha)) if (alpha != 1.0 and rf > 0) else float("nan")
    final = {"seed": seed, "alpha": alpha, "arm": arm, "diverged": False,
             "acc_av": te["acc_av"], "acc_audio": te["acc_a"], "acc_video": te["acc_v"],
             "delta_audio": te["delta_a"], "delta_video": te["delta_v"],
             "dominance": te["dominance"], "abs_imbalance": abs(te["delta_a"] - te["delta_v"]),
             "multimodal_gain": te["multimodal_gain"], "f1_av": te["f1_av"],
             "rho_enc": re, "rho_fusion": rf, "compensation": comp,
             "grad_ratio_final": hist[-1]["grad_ratio"], "update_ratio_final": hist[-1]["update_ratio"],
             "runtime_s": round(time.time() - t0, 1)}
    return final, hist


def sanity_check(cfg, feats, device):
    res = {"scalar": {}, "checks": {}, "pass": True}
    alpha = 10.0
    for arm in ARMS:
        h = torch.tensor([[2.0]], requires_grad=True, device=device)
        out = arm_apply(h, alpha, arm); out.sum().backward()
        fwd, bwd = float(out.item() / 2.0), float(h.grad.item())
        exp = {"baseline": (1, 1), "forward": (alpha, 1), "gradient": (1, alpha), "ordinary": (alpha, alpha)}[arm]
        ok = abs(fwd - exp[0]) < 1e-6 and abs(bwd - exp[1]) < 1e-6
        res["scalar"][arm] = {"forward_mult": fwd, "backward_mult": bwd, "pass": bool(ok)}; res["pass"] &= ok
    # CREMA-D model + real batch
    set_seed(0, True)
    m = CremadMultimodalModel(cfg["data"]["n_classes"], rep_dim=cfg["model"]["rep_dim"],
                              fusion=cfg["model"]["fusion"], normalization="none", head=cfg["model"]["head"]).to(device)
    a_tr, v_tr, y_tr = feats["train"]; a, v, y = a_tr[:64], v_tr[:64], y_tr[:64]
    lf = nn.CrossEntropyLoss()
    with torch.no_grad():
        hb, _ = _reps(m, a, v); base_post = arm_apply(hb, alpha, "baseline")
        hg, _ = _reps(m, a, v); grad_post = arm_apply(hg, alpha, "gradient")
    res["checks"]["gradient_only_forward_identical_to_baseline"] = bool(torch.equal(base_post, grad_post))

    def enc_a_grad(arm):
        m.zero_grad(set_to_none=True)
        lf(arm_logits(m, a, v, alpha, arm), y).backward()
        return math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.enc_a.parameters()))
    gb_ = enc_a_grad("baseline"); gf = enc_a_grad("forward"); gg = enc_a_grad("gradient"); go = enc_a_grad("ordinary")
    res["checks"]["enc_a_grad_norm"] = {"baseline": gb_, "forward": gf, "gradient": gg, "ordinary": go}
    res["checks"]["forward_ratio_vs_baseline"] = gf / gb_ if gb_ > 0 else float("nan")
    res["checks"]["gradient_ratio_vs_baseline"] = gg / gb_ if gb_ > 0 else float("nan")
    res["pass"] = bool(res["pass"] and res["checks"]["gradient_only_forward_identical_to_baseline"])
    return res


def _ci(v):
    v = np.asarray(v, float); sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return float(np.mean(v)), sd, (1.96 * sd / math.sqrt(len(v)) if len(v) > 1 else 0.0)


def _paired(fin, alpha, arm1, arm2, col="dominance"):
    a = fin[(fin.alpha == alpha) & (fin.arm == arm1) & (~fin.diverged)].sort_values("seed")[col].values
    b = fin[(fin.alpha == alpha) & (fin.arm == arm2) & (~fin.diverged)].sort_values("seed")[col].values
    d = b - a; sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    t = sps.ttest_rel(b, a) if len(d) > 1 and sd > 0 else None
    return {"mean_diff": float(np.mean(d)), "ci95": (1.96 * sd / math.sqrt(len(d))) if len(d) > 1 else 0.0,
            "cohen_d": (float(np.mean(d)) / sd) if sd > 0 else float("nan"),
            "p": float(t.pvalue) if t else float("nan")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = load_features(device)
    out = ROOT / "results" / "cremad_disentangle"; out.mkdir(parents=True, exist_ok=True)

    san = sanity_check(cfg, feats, device)
    json.dump(san, open(out / "cremad_disentangle_sanity.json", "w"), indent=2, default=float)
    print("=== AUTOGRAD SANITY (CREMA-D model + real features) ===")
    for arm in ARMS:
        s = san["scalar"][arm]; print(f"  {arm:9s} fwd={s['forward_mult']:.4g} bwd={s['backward_mult']:.4g} pass={s['pass']}")
    print("  grad-only forward==baseline:", san["checks"]["gradient_only_forward_identical_to_baseline"])
    print("  enc_A grad ratio vs baseline: forward=%.3f gradient=%.3f" % (
        san["checks"]["forward_ratio_vs_baseline"], san["checks"]["gradient_ratio_vs_baseline"]))
    print("  SANITY PASS:", san["pass"])
    if not san["pass"]:
        print("SANITY GATE FAILED — stopping."); return 1
    if args.sanity_only:
        print("sanity-only: gate passed, stopping."); return 0

    seeds = cfg["seed_policy"]["seeds"]
    finals, trajs = [], []; t0 = time.time()
    for alpha in ALPHAS:
        for arm in ARMS:
            for seed in seeds:
                final, hist = train_one(cfg, feats, device, seed, alpha, arm)
                finals.append(final)
                for h in hist:
                    trajs.append({"alpha": alpha, "arm": arm, "seed": seed, **h})
            g = [f for f in finals if f["alpha"] == alpha and f["arm"] == arm and not f.get("diverged")]
            dom = np.mean([f["dominance"] for f in g]) if g else float("nan")
            print(f"  alpha={alpha:g} {arm:9s} dom={dom:+.3f}  ({time.time()-t0:.0f}s)")

    fin = pd.DataFrame(finals); tj = pd.DataFrame(trajs)
    fin.to_csv(out / "cremad_disentangle_primary.csv", index=False)
    tj.to_csv(out / "cremad_disentangle_trajectories.csv", index=False)
    json.dump({"config": cfg, "git_commit": git_hash(), "seeds": seeds, "alphas": ALPHAS, "arms": ARMS,
               "modality_scaled": "audio", "optimizer": "adam"},
              open(out / "cremad_disentangle_config.json", "w"), indent=2, default=str)
    analyze(fin, tj, out)
    return 0


def analyze(fin, tj, out):
    ok = fin[~fin.diverged]
    rows = []
    for alpha in ALPHAS:
        for arm in ARMS:
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            for col in ["dominance", "multimodal_gain", "acc_av", "delta_audio", "delta_video",
                        "rho_enc", "rho_fusion", "grad_ratio_final", "update_ratio_final"]:
                m, sd, ci = _ci(g[col].values)
                rows.append({"alpha": alpha, "arm": arm, "metric": col, "mean": m, "sd": sd, "ci95": ci})
    pd.DataFrame(rows).to_csv(out / "cremad_disentangle_agg.csv", index=False)

    a = {"n_diverged": int(fin.diverged.sum()), "by_alpha": {}}
    for alpha in ALPHAS:
        base = float(ok[(ok.alpha == alpha) & (ok.arm == "baseline")].dominance.mean())
        block = {"baseline_dominance": base}
        for arm in ARMS:
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            block[arm] = {"dominance": m, "sd": sd, "ci95": ci, "shift_vs_baseline": m - base,
                          "gain": float(g.multimodal_gain.mean()), "acc_av": float(g.acc_av.mean()),
                          "rho_enc": float(g.rho_enc.mean()), "rho_fusion": float(g.rho_fusion.mean()),
                          "grad_ratio": float(g.grad_ratio_final.mean()), "update_ratio": float(g.update_ratio_final.mean())}
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
        case = "CASE_null — no arm shifts dominance (fully compensated / degenerate)"
    elif fs > thr and gs < thr:
        case = "CASE_A — forward-dominated (matches synthetic Adam)"
    elif gs > thr and fs < thr:
        case = "CASE_B — gradient-dominated"
    elif fs > thr and gs > thr:
        case = "CASE_C — mixed"
    else:
        case = "CASE_D/mixed"
    a["classification"] = case
    a["seed_sd_alpha100"] = {arm: float(ok[(ok.alpha == 100.0) & (ok.arm == arm)].dominance.std(ddof=1)) for arm in ARMS}
    json.dump(a, open(out / "cremad_disentangle_analysis.json", "w"), indent=2, default=float)

    print("\n=== CREMA-D dominance (Acc_audio-Acc_video) by arm ===")
    for alpha in ALPHAS:
        print(f" alpha={alpha:g}: baseline dom={a['by_alpha'][alpha]['baseline_dominance']:+.3f}")
        for arm in ARMS:
            x = a["by_alpha"][alpha][arm]
            print(f"   {arm:9s} dom={x['dominance']:+.3f} shift={x['shift_vs_baseline']:+.3f} gain={x['gain']:+.3f} "
                  f"rho_enc={x['rho_enc']:.2f} rho_fus={x['rho_fusion']:.2f} grad_r={x['grad_ratio']:.1f} upd_r={x['update_ratio']:.2f}")
    print("CLASSIFICATION:", a["classification"])
    figures(fin, tj, a, out)


def figures(fin, tj, a, out):
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    ok = fin[~fin.diverged]
    COL = {"baseline": "#000000", "forward": "#ff7f0e", "gradient": "#1f77b4", "ordinary": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, alpha in zip(axes, ALPHAS):
        base = a["by_alpha"][alpha]["baseline_dominance"]
        for j, arm in enumerate(ARMS):
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            ax.bar(j, m - base, yerr=ci, capsize=3, color=COL[arm], alpha=0.8)
            ax.scatter([j] * len(g), g.dominance.values - base, color="0.3", s=14, zorder=3)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_title(f"CREMA-D alpha={alpha:g}: dominance shift vs baseline"); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("CREMA-D forward vs gradient pathway (audio scaled, Adam, 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_dis_dominance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, tit in zip(axes, ["grad_ratio_final", "update_ratio_final"], ["encoder gradient ratio", "parameter-update ratio"]):
        for j, arm in enumerate(ARMS):
            g = ok[(ok.alpha == 100.0) & (ok.arm == arm)]
            m, sd, ci = _ci(g[key].values); ax.bar(j, m, yerr=ci, capsize=3, color=COL[arm])
            ax.scatter([j] * len(g), g[key].values, color="0.3", s=12, zorder=3)
        ax.axhline(1, color="0.6", ls=":"); ax.set_yscale("log"); ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_title(f"CREMA-D alpha=100: {tit} audio/video"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_dis_grad_update_ratio.png", dpi=150); plt.close(fig)

    tn = tj[tj.alpha == 100.0]
    fig, axx = plt.subplots(1, 4, figsize=(18, 4.2))
    for k, key in enumerate(["val_dominance", "val_gain", "rho_fusion", "rho_enc"]):
        for arm in ARMS:
            g = tn[tn.arm == arm]
            if len(g):
                m = g.groupby("epoch")[key].mean(); axx[k].plot(m.index, m.values, color=COL[arm], label=arm, lw=1.7)
        axx[k].set_title(key); axx[k].set_xlabel("epoch"); axx[k].grid(alpha=0.25); axx[k].legend(fontsize=7)
        if key in ("rho_fusion", "rho_enc"):
            axx[k].set_yscale("log"); axx[k].axhline(1, color="0.6", ls=":")
    fig.suptitle("CREMA-D alpha=100 trajectories by arm (mean of 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "cremad_dis_trajectories.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
