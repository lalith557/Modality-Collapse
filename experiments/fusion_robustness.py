"""Fusion-mechanism robustness of the forward-vs-gradient decomposition (Phase 3).
See PREREGISTRATION_FUSION_ROBUSTNESS.md.

Tests whether forward-only ≈ ordinary and gradient-only ≈ baseline (the Adam decomposition) survive
sum / concat / learned-gated fusion. Reuses the synthetic generator, TinyMLPEncoder, and validated
ForwardScale/GradScale. Frozen models/fusion.py + multimodal_model.py are NOT edited (local model here).

Usage:
    python experiments/fusion_robustness.py --sanity-only
    python experiments/fusion_robustness.py
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
from experiments.disentangle_scale_grad import arm_apply
from experiments.train import _to_device
from models.encoders import TinyMLPEncoder
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ARMS = ["baseline", "forward", "gradient", "ordinary"]
FUSIONS = ["sum", "concat", "gated"]
ALPHA = 100.0


class FusionModel(nn.Module):
    """Two-stream model with selectable fusion; arm applied to modality A at the fusion boundary."""
    def __init__(self, in_a, in_v, n_classes, hidden=64, rep=32, fusion="sum"):
        super().__init__()
        self.fusion = fusion; self.rep = rep
        self.enc_a = TinyMLPEncoder(in_a, hidden, rep)
        self.enc_v = TinyMLPEncoder(in_v, hidden, rep)
        if fusion == "gated":
            self.gate_a = nn.Linear(rep, rep); self.gate_v = nn.Linear(rep, rep)
        fdim = 2 * rep if fusion == "concat" else rep
        self.head = nn.Linear(fdim, n_classes)

    def reps(self, x_a, x_v):
        return self.enc_a(x_a), self.enc_v(x_v)

    def fuse_arm(self, h_a, h_v, alpha, arm, mask_a=1.0, mask_v=1.0):
        a = arm_apply(h_a, alpha, arm)
        if self.fusion == "gated":
            ga = torch.sigmoid(self.gate_a(h_a)); gv = torch.sigmoid(self.gate_v(h_v))
            a = ga * a; v = gv * h_v
        else:
            v = h_v
        a = mask_a * a; v = mask_v * v
        if self.fusion == "concat":
            return torch.cat([a, v], dim=1)
        return a + v

    def forward(self, x_a, x_v, alpha, arm, mask_a=1.0, mask_v=1.0):
        h_a, h_v = self.reps(x_a, x_v)
        return self.head(self.fuse_arm(h_a, h_v, alpha, arm, mask_a, mask_v))


@torch.no_grad()
def contrib(model, a, v, y, alpha, arm):
    was = model.training; model.eval()
    acc = {}
    for name, (ma, mv) in {"both": (1.0, 1.0), "a": (1.0, 0.0), "v": (0.0, 1.0)}.items():
        acc[name] = float((model(a, v, alpha, arm, ma, mv).argmax(1) == y).float().mean())
    if was:
        model.train()
    return {"acc_av": acc["both"], "acc_a": acc["a"], "acc_v": acc["v"],
            "dominance": acc["a"] - acc["v"], "multimodal_gain": acc["both"] - max(acc["a"], acc["v"])}


@torch.no_grad()
def rhos(model, a, v, alpha, arm):
    was = model.training; model.eval()
    h_a, h_v = model.reps(a, v); h_post = arm_apply(h_a, alpha, arm)
    re = float(h_a.norm(dim=1).mean() / h_v.norm(dim=1).mean())
    rf = float(h_post.norm(dim=1).mean() / h_v.norm(dim=1).mean())
    if was:
        model.train()
    return re, rf


def train_one(cfg, ds, device, seed, fusion, arm):
    set_seed(seed, True)
    nc = ds.meta["n_classes"]; data = _to_device(ds, device)
    a_tr, v_tr, y_tr = data["train"]; a_te, v_te, y_te = data["test"]
    model = FusionModel(ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc, fusion=fusion).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=0.0)
    lf = nn.CrossEntropyLoss()
    epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    last = {}; diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); ga = gb = ua = ub = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lf(model(a_tr[idx], v_tr[idx], ALPHA, arm), y_tr[idx])
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in ea if p.grad is not None) ** 0.5
            gb += sum(float(p.grad.pow(2).sum()) for p in ev if p.grad is not None) ** 0.5
            pa = [p.detach().clone() for p in ea]; pv = [p.detach().clone() for p in ev]
            opt.step()
            ua += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ea, pa)) ** 0.5
            ub += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ev, pv)) ** 0.5
            nb += 1
        if diverged:
            break
        last = {"grad_ratio": (ga / gb) if gb > 0 else float("nan"), "update_ratio": (ua / ub) if ub > 0 else float("nan")}
    if diverged:
        return {"seed": seed, "fusion": fusion, "arm": arm, "diverged": True}
    te = contrib(model, a_te, v_te, y_te, ALPHA, arm); re, rf = rhos(model, a_te, v_te, ALPHA, arm)
    comp = (1.0 - math.log(rf) / math.log(ALPHA)) if rf > 0 else float("nan")
    return {"seed": seed, "fusion": fusion, "arm": arm, "diverged": False,
            "acc_av": te["acc_av"], "dominance": te["dominance"], "multimodal_gain": te["multimodal_gain"],
            "rho_enc": re, "rho_fusion": rf, "compensation": comp,
            "grad_ratio_final": last["grad_ratio"], "update_ratio_final": last["update_ratio"]}


def sanity_check(device):
    ok = True
    for arm in ARMS:
        h = torch.tensor([[2.0]], requires_grad=True, device=device)
        out = arm_apply(h, 10.0, arm); out.sum().backward()
        exp = {"baseline": (1, 1), "forward": (10, 1), "gradient": (1, 10), "ordinary": (10, 10)}[arm]
        ok &= abs(out.item() / 2.0 - exp[0]) < 1e-6 and abs(float(h.grad.item()) - exp[1]) < 1e-6
    return {"pass": bool(ok)}


def _ci(v):
    v = np.asarray(v, float); sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return float(np.mean(v)), sd, (1.96 * sd / math.sqrt(len(v)) if len(v) > 1 else 0.0)


def _paired(fin, fusion, arm1, arm2, col="dominance"):
    a = fin[(fin.fusion == fusion) & (fin.arm == arm1) & (~fin.diverged)].sort_values("seed")[col].values
    b = fin[(fin.fusion == fusion) & (fin.arm == arm2) & (~fin.diverged)].sort_values("seed")[col].values
    d = b - a; sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    t = sps.ttest_rel(b, a) if len(d) > 1 and sd > 0 else None
    return {"mean_diff": float(np.mean(d)), "ci95": (1.96 * sd / math.sqrt(len(d))) if len(d) > 1 else 0.0,
            "cohen_d": (float(np.mean(d)) / sd) if sd > 0 else float("nan"), "p": float(t.pvalue) if t else float("nan")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    out = ROOT / "results" / "fusion_robustness"; out.mkdir(parents=True, exist_ok=True)
    san = sanity_check(device); print("SANITY PASS:", san["pass"])
    if not san["pass"]:
        print("SANITY FAILED — stopping."); return 1
    if args.sanity_only:
        return 0

    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    seeds = cfg["seed_policy"]["seeds"]; finals = []; t0 = time.time()
    for fusion in FUSIONS:
        for arm in ARMS:
            for seed in seeds:
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
                ds = generate_dataset(p)
                finals.append(train_one(cfg, ds, device, seed, fusion, arm))
            g = [f for f in finals if f["fusion"] == fusion and f["arm"] == arm and not f.get("diverged")]
            dom = np.mean([f["dominance"] for f in g]) if g else float("nan")
            print(f"  {fusion:7s} {arm:9s} dom={dom:+.3f}  ({time.time()-t0:.0f}s)")

    fin = pd.DataFrame(finals); fin.to_csv(out / "fusion_robustness_primary.csv", index=False)
    json.dump({"config": cfg, "git_commit": git_hash(), "seeds": seeds, "alpha": ALPHA,
               "arms": ARMS, "fusions": FUSIONS, "optimizer": "adam"},
              open(out / "fusion_robustness_config.json", "w"), indent=2, default=str)
    analyze(fin, out)
    return 0


def analyze(fin, out):
    ok = fin[~fin.diverged]; rows = []
    for fusion in FUSIONS:
        for arm in ARMS:
            g = ok[(ok.fusion == fusion) & (ok.arm == arm)]
            for col in ["dominance", "multimodal_gain", "acc_av", "rho_enc", "rho_fusion", "grad_ratio_final", "update_ratio_final"]:
                m, sd, ci = _ci(g[col].values)
                rows.append({"fusion": fusion, "arm": arm, "metric": col, "mean": m, "sd": sd, "ci95": ci})
    pd.DataFrame(rows).to_csv(out / "fusion_robustness_agg.csv", index=False)

    a = {"n_diverged": int(fin.diverged.sum()), "by_fusion": {}}
    for fusion in FUSIONS:
        base = float(ok[(ok.fusion == fusion) & (ok.arm == "baseline")].dominance.mean())
        block = {"baseline_dominance": base}
        for arm in ARMS:
            g = ok[(ok.fusion == fusion) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            block[arm] = {"dominance": m, "shift_vs_baseline": m - base, "ci95": ci,
                          "gain": float(g.multimodal_gain.mean()), "acc_av": float(g.acc_av.mean()),
                          "rho_enc": float(g.rho_enc.mean()), "grad_ratio": float(g.grad_ratio_final.mean()),
                          "update_ratio": float(g.update_ratio_final.mean())}
        block["cmp"] = {"forward_vs_baseline": _paired(fin, fusion, "baseline", "forward"),
                        "gradient_vs_baseline": _paired(fin, fusion, "baseline", "gradient"),
                        "ordinary_vs_forward": _paired(fin, fusion, "forward", "ordinary")}
        fs = abs(block["cmp"]["forward_vs_baseline"]["mean_diff"]); gs = abs(block["cmp"]["gradient_vs_baseline"]["mean_diff"])
        block["case"] = ("CASE_A forward-dominated" if fs > 0.02 and gs < 0.02 else
                         "CASE_B gradient-dominated" if gs > 0.02 and fs < 0.02 else
                         "CASE_C mixed" if fs > 0.02 and gs > 0.02 else "CASE_null")
        a["by_fusion"][fusion] = block
    json.dump(a, open(out / "fusion_robustness_analysis.json", "w"), indent=2, default=float)

    print("\n=== fusion robustness: dominance shift by arm ===")
    for fusion in FUSIONS:
        bl = a["by_fusion"][fusion]["baseline_dominance"]
        print(f" {fusion}: baseline dom={bl:+.3f}  [{a['by_fusion'][fusion]['case']}]")
        for arm in ARMS:
            x = a["by_fusion"][fusion][arm]
            print(f"   {arm:9s} dom={x['dominance']:+.3f} shift={x['shift_vs_baseline']:+.3f} gain={x['gain']:+.3f} "
                  f"grad_r={x['grad_ratio']:.1f} upd_r={x['update_ratio']:.2f}")
    figures(fin, a, out)


def figures(fin, a, out):
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    ok = fin[~fin.diverged]
    COL = {"baseline": "#000000", "forward": "#ff7f0e", "gradient": "#1f77b4", "ordinary": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, fusion in zip(axes, FUSIONS):
        base = a["by_fusion"][fusion]["baseline_dominance"]
        for j, arm in enumerate(ARMS):
            g = ok[(ok.fusion == fusion) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            ax.bar(j, m - base, yerr=ci, capsize=3, color=COL[arm])
            ax.scatter([j] * len(g), g.dominance.values - base, color="0.3", s=12, zorder=3)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, fontsize=8)
        ax.set_title(f"{fusion} fusion  [{a['by_fusion'][fusion]['case'].split()[0]}]"); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Fusion-mechanism robustness: dominance shift vs baseline (alpha=100, Adam, 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "fusion_dominance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, tit in zip(axes, ["grad_ratio_final", "update_ratio_final"], ["gradient ratio", "update ratio"]):
        x = np.arange(len(FUSIONS)); w = 0.2
        for j, arm in enumerate(ARMS):
            vals = [_ci(ok[(ok.fusion == f) & (ok.arm == arm)][key].values)[0] for f in FUSIONS]
            ax.bar(x + (j - 1.5) * w, vals, w, label=arm, color=COL[arm])
        ax.set_yscale("log"); ax.axhline(1, color="0.6", ls=":"); ax.set_xticks(x); ax.set_xticklabels(FUSIONS)
        ax.set_title(f"alpha=100 {tit} A/V by fusion"); ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "fusion_grad_update_ratio.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
