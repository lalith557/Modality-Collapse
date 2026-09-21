"""Forward-scale vs gradient-scale disentangling control. See PREREGISTRATION_SCALE_GRADIENT_DISENTANGLE.md.

Separates the two pathways coupled by the ordinary training-time scale h_A' = alpha*h_A:
  baseline      : forward x1, backward x1
  forward-only  : forward xalpha, backward x1   (alpha removed from encoder gradient)
  gradient-only : forward x1,     backward xalpha (encoder gradient scaled, forward unchanged)
  ordinary      : forward xalpha, backward xalpha (standard multiply)

Reuses the E4/E5 equal-information generator/architecture/optimizer UNCHANGED. Prior experiments frozen.

Usage:
    python experiments/disentangle_scale_grad.py --sanity-only
    python experiments/disentangle_scale_grad.py
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
from experiments.train import _optimizer, _to_device
from models.fusion import fuse
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ARMS = ["baseline", "forward", "gradient", "ordinary"]
ALPHAS = [1.0, 10.0, 100.0]


# --------------------------------------------------------------------------- #
# Autograd: the two isolating operations
# --------------------------------------------------------------------------- #
class ForwardScale(torch.autograd.Function):
    """value = alpha*x ; d value / d x = 1  (forward magnitude changes, gradient NOT scaled)."""
    @staticmethod
    def forward(ctx, x, alpha):
        return alpha * x

    @staticmethod
    def backward(ctx, g):
        return g, None            # pass gradient through unchanged (x1)


class GradScale(torch.autograd.Function):
    """value = x ; d value / d x = alpha  (forward unchanged, encoder gradient scaled)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return g * ctx.alpha, None


def arm_apply(h, alpha, arm):
    if arm == "baseline" or alpha == 1.0:
        return h
    if arm == "forward":
        return ForwardScale.apply(h, alpha)
    if arm == "gradient":
        return GradScale.apply(h, alpha)
    if arm == "ordinary":
        return alpha * h
    raise ValueError(arm)


def _reps(model, x_a, x_v):
    """Fusion-boundary reps BEFORE the arm transform (normalization='none')."""
    return model.norm_a(model.enc_a(x_a)), model.norm_v(model.enc_v(x_v))


def arm_forward(model, x_a, x_v, alpha, arm, mask_a=1.0, mask_v=1.0):
    h_a, h_v = _reps(model, x_a, x_v)
    h_a_post = arm_apply(h_a, alpha, arm)
    z = fuse(model.fusion, mask_a * h_a_post, mask_v * h_v)
    return model.head(z), h_a, h_v, h_a_post


# --------------------------------------------------------------------------- #
# Autograd sanity gate
# --------------------------------------------------------------------------- #
def sanity_check(device):
    """Scalar forward/backward multiplier verification + numeric activation/gradient checks."""
    res = {"scalar": {}, "checks": {}, "pass": True}
    alpha = 10.0
    for arm in ARMS:
        h = torch.tensor([[2.0]], requires_grad=True, device=device)
        out = arm_apply(h, alpha, arm)
        loss = out.sum()
        loss.backward()
        fwd_mult = float(out.item() / 2.0)
        bwd_mult = float(h.grad.item())      # baseline dL/dh = 1
        exp = {"baseline": (1, 1), "forward": (alpha, 1), "gradient": (1, alpha), "ordinary": (alpha, alpha)}[arm]
        ok = abs(fwd_mult - exp[0]) < 1e-6 and abs(bwd_mult - exp[1]) < 1e-6
        res["scalar"][arm] = {"forward_mult": fwd_mult, "backward_mult": bwd_mult,
                              "expected_forward": exp[0], "expected_backward": exp[1], "pass": bool(ok)}
        res["pass"] &= ok

    # Numeric checks on a real tiny model: (i) gradient-only forward == baseline forward (bit-identical);
    # (ii) forward-only encoder gradient does NOT carry alpha; (iii) ordinary carries both.
    set_seed(0, True)
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    m = build_model(cfg, 32, 32, 8).to(device)
    xa = torch.randn(64, 32, device=device); xv = torch.randn(64, 32, device=device)
    y = torch.randint(0, 8, (64,), device=device)
    lf = nn.CrossEntropyLoss()

    def enc_a_grad(arm):
        m.zero_grad(set_to_none=True)
        logits, h_a, h_v, h_post = arm_forward(m, xa, xv, alpha, arm)
        lf(logits, y).backward()
        g = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.enc_a.parameters()))
        return g, h_post.detach().clone()

    with torch.no_grad():
        _, _, _, base_post = arm_forward(m, xa, xv, alpha, "baseline")
        _, _, _, grad_post = arm_forward(m, xa, xv, alpha, "gradient")
    ident = torch.allclose(base_post, grad_post, atol=0, rtol=0)
    res["checks"]["gradient_only_forward_identical_to_baseline"] = bool(ident)

    g_base, _ = enc_a_grad("baseline")
    g_fwd, _ = enc_a_grad("forward")
    g_grad, _ = enc_a_grad("gradient")
    g_ord, _ = enc_a_grad("ordinary")
    res["checks"]["enc_a_grad_norm"] = {"baseline": g_base, "forward": g_fwd, "gradient": g_grad, "ordinary": g_ord}
    # forward-only must NOT scale the enc gradient by alpha (ratio to baseline far below alpha);
    # gradient-only and ordinary DO scale it (ratio ~alpha, up to nonlinearity).
    res["checks"]["forward_ratio_vs_baseline"] = g_fwd / g_base if g_base > 0 else float("nan")
    res["checks"]["gradient_ratio_vs_baseline"] = g_grad / g_base if g_base > 0 else float("nan")
    res["checks"]["ordinary_ratio_vs_baseline"] = g_ord / g_base if g_base > 0 else float("nan")
    res["pass"] = bool(res["pass"] and ident)
    return res


# --------------------------------------------------------------------------- #
# Metrics using the arm forward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def arm_contrib(model, x_a, x_v, y, alpha, arm, nc):
    was = model.training; model.eval()
    accs = {}
    for name, (ma, mv) in {"both": (1.0, 1.0), "a": (1.0, 0.0), "v": (0.0, 1.0)}.items():
        logits, *_ = arm_forward(model, x_a, x_v, alpha, arm, ma, mv)
        accs[name] = float((logits.argmax(1) == y).float().mean())
    if was:
        model.train()
    acc_av, acc_a, acc_v = accs["both"], accs["a"], accs["v"]
    return {"acc_av": acc_av, "acc_a": acc_a, "acc_v": acc_v,
            "delta_a": acc_av - acc_v, "delta_v": acc_av - acc_a,
            "multimodal_gain": acc_av - max(acc_a, acc_v), "dominance": acc_a - acc_v}


@torch.no_grad()
def arm_rho(model, x_a, x_v, alpha, arm):
    was = model.training; model.eval()
    h_a, h_v = _reps(model, x_a, x_v)
    h_post = arm_apply(h_a, alpha, arm)
    rho_enc = float(h_a.norm(dim=1).mean() / h_v.norm(dim=1).mean())
    rho_fus = float(h_post.norm(dim=1).mean() / h_v.norm(dim=1).mean())
    if was:
        model.train()
    return rho_enc, rho_fus


def train_one(cfg, ds, device, seed, alpha, arm):
    set_seed(seed, True)
    nc = ds.meta["n_classes"]
    data = _to_device(ds, device)
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]; a_te, v_te, y_te = data["test"]
    model = build_model(cfg, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    opt = _optimizer(model, cfg, device)
    lf = nn.CrossEntropyLoss()
    epochs = int(cfg["train"]["epochs"]); bs = int(cfg["train"]["batch_size"]); n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    hist = []; diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); ga = gb = 0.0; nb = 0
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
            opt.step(); nb += 1
        if diverged:
            break
        re, rf = arm_rho(model, a_va, v_va, alpha, arm)
        c = arm_contrib(model, a_va, v_va, y_va, alpha, arm, nc)
        hist.append({"epoch": ep, "rho_enc": re, "rho_fusion": rf,
                     "val_dominance": c["dominance"], "val_gain": c["multimodal_gain"],
                     "val_acc_av": c["acc_av"], "grad_ratio": (ga / gb) if gb > 0 else float("nan"),
                     "grad_a": ga / nb, "grad_v": gb / nb})
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
             "grad_ratio_final": hist[-1]["grad_ratio"]}
    return final, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()
    device = get_device({"type": "cuda", "allow_cpu": False})
    print("Device:", device_report(device))
    out = ROOT / "results" / "disentangle"; out.mkdir(parents=True, exist_ok=True)

    san = sanity_check(device)
    json.dump(san, open(out / "disentangle_sanity.json", "w"), indent=2, default=float)
    print("=== AUTOGRAD SANITY ===")
    for arm in ARMS:
        s = san["scalar"][arm]
        print(f"  {arm:9s} fwd={s['forward_mult']:.4g}(exp {s['expected_forward']}) "
              f"bwd={s['backward_mult']:.4g}(exp {s['expected_backward']}) pass={s['pass']}")
    print("  gradient-only forward == baseline (bit-identical):", san["checks"]["gradient_only_forward_identical_to_baseline"])
    print("  enc_A grad ratio vs baseline: forward=%.3f gradient=%.3f ordinary=%.3f" % (
        san["checks"]["forward_ratio_vs_baseline"], san["checks"]["gradient_ratio_vs_baseline"],
        san["checks"]["ordinary_ratio_vs_baseline"]))
    print("  SANITY PASS:", san["pass"])
    if not san["pass"]:
        print("SANITY GATE FAILED — stopping, not running the main experiment.")
        return 1
    if args.sanity_only:
        print("sanity-only: stopping before main run (gate passed).")
        return 0

    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    seeds = cfg["seed_policy"]["seeds"]
    finals, trajs = [], []
    t0 = time.time()
    for alpha in ALPHAS:
        for arm in ARMS:
            for seed in seeds:
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)  # unscaled data
                ds = generate_dataset(p)
                final, hist = train_one(cfg, ds, device, seed, alpha, arm)
                finals.append(final)
                for h in hist:
                    trajs.append({"alpha": alpha, "arm": arm, "seed": seed, **h})
            g = [f for f in finals if f["alpha"] == alpha and f["arm"] == arm and not f.get("diverged")]
            dom = np.mean([f["dominance"] for f in g]) if g else float("nan")
            print(f"  alpha={alpha:g} {arm:9s} dom={dom:+.3f}  ({time.time()-t0:.0f}s)")

    fin = pd.DataFrame(finals); tj = pd.DataFrame(trajs)
    fin.to_csv(out / "disentangle_primary.csv", index=False)
    tj.to_csv(out / "disentangle_trajectories.csv", index=False)
    json.dump({"config": cfg, "git_commit": git_hash(), "seeds": seeds, "alphas": ALPHAS, "arms": ARMS},
              open(out / "disentangle_config.json", "w"), indent=2, default=str)
    analyze(fin, tj, out)
    return 0


def _ci(v):
    v = np.asarray(v, float)
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return float(np.mean(v)), sd, (1.96 * sd / math.sqrt(len(v)) if len(v) > 1 else 0.0)


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


def analyze(fin, tj, out):
    ok = fin[~fin.diverged]
    rows = []
    for alpha in ALPHAS:
        for arm in ARMS:
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            for col in ["dominance", "multimodal_gain", "acc_av", "rho_enc", "rho_fusion", "delta_a", "delta_v"]:
                m, sd, ci = _ci(g[col].values)
                rows.append({"alpha": alpha, "arm": arm, "metric": col, "mean": m, "sd": sd, "ci95": ci})
    pd.DataFrame(rows).to_csv(out / "disentangle_agg.csv", index=False)

    a = {"n_diverged": int(fin.diverged.sum()), "by_alpha": {}}
    base_by_a = {al: float(ok[(ok.alpha == al) & (ok.arm == "baseline")].dominance.mean()) for al in ALPHAS}
    for alpha in [10.0, 100.0]:
        base = base_by_a[alpha]
        block = {"baseline_dominance": base}
        for arm in ARMS:
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            block[arm] = {"dominance": m, "sd": sd, "ci95": ci,
                          "shift_vs_baseline": m - base,
                          "gain": float(g.multimodal_gain.mean()), "acc_av": float(g.acc_av.mean()),
                          "rho_enc": float(g.rho_enc.mean()), "rho_fusion": float(g.rho_fusion.mean())}
        block["cmp"] = {
            "forward_vs_baseline": _paired(fin, alpha, "baseline", "forward"),
            "gradient_vs_baseline": _paired(fin, alpha, "baseline", "gradient"),
            "ordinary_vs_baseline": _paired(fin, alpha, "baseline", "ordinary"),
            "ordinary_vs_forward": _paired(fin, alpha, "forward", "ordinary"),
            "ordinary_vs_gradient": _paired(fin, alpha, "gradient", "ordinary"),
        }
        a["by_alpha"][alpha] = block

    # classification at alpha=100 (largest signal)
    b = a["by_alpha"][100.0]
    ord_shift = abs(b["cmp"]["ordinary_vs_baseline"]["mean_diff"])
    fwd_shift = abs(b["cmp"]["forward_vs_baseline"]["mean_diff"])
    grd_shift = abs(b["cmp"]["gradient_vs_baseline"]["mean_diff"])
    thr = 0.02
    # is ordinary itself an effect at all?
    if ord_shift < thr and fwd_shift < thr and grd_shift < thr:
        case = "CASE_null — no isolable contribution shift in any arm (ordinary itself ~compensated)"
    else:
        fwd_like_ord = abs(fwd_shift - ord_shift) < 0.5 * ord_shift + 1e-9 and fwd_shift > thr
        grd_like_ord = abs(grd_shift - ord_shift) < 0.5 * ord_shift + 1e-9 and grd_shift > thr
        fwd_like_base = fwd_shift < thr
        grd_like_base = grd_shift < thr
        if fwd_like_ord and grd_like_base:
            case = "CASE_A — forward-dominated"
        elif grd_like_ord and fwd_like_base:
            case = "CASE_B — gradient-dominated"
        elif fwd_shift > thr and grd_shift > thr:
            case = "CASE_C — mixed (both pathways contribute)"
        elif ord_shift > thr and fwd_shift < thr and grd_shift < thr:
            case = "CASE_D — interactive (neither isolated arm reproduces ordinary)"
        else:
            case = "CASE_C — mixed/partial"
    # seed stability check
    sd100 = {arm: float(ok[(ok.alpha == 100.0) & (ok.arm == arm)].dominance.std(ddof=1)) for arm in ARMS}
    if max(sd100.values()) > 0.15:
        case += " | NOTE high seed variance (CASE_E risk)"
    a["classification"] = case
    a["seed_sd_alpha100"] = sd100
    json.dump(a, open(out / "disentangle_analysis.json", "w"), indent=2, default=float)

    print("\n=== dominance by arm (mean, shift vs baseline) ===")
    for alpha in [10.0, 100.0]:
        print(f" alpha={alpha:g}: baseline dom={a['by_alpha'][alpha]['baseline_dominance']:+.3f}")
        for arm in ARMS:
            x = a["by_alpha"][alpha][arm]
            print(f"   {arm:9s} dom={x['dominance']:+.3f} shift={x['shift_vs_baseline']:+.3f} "
                  f"gain={x['gain']:+.3f} rho_enc={x['rho_enc']:.2f} rho_fus={x['rho_fusion']:.2f}")
    print("CLASSIFICATION:", a["classification"])
    figures(fin, tj, a, out)


def figures(fin, tj, a, out):
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    ok = fin[~fin.diverged]
    COL = {"baseline": "#000000", "forward": "#ff7f0e", "gradient": "#1f77b4", "ordinary": "#d62728"}

    # Fig 1: dominance shift vs baseline, by arm, for alpha 10 & 100 (bars + per-seed points)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, alpha in zip(axes, [10.0, 100.0]):
        base = a["by_alpha"][alpha]["baseline_dominance"]
        for j, arm in enumerate(ARMS):
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            m, sd, ci = _ci(g.dominance.values)
            ax.bar(j, m - base, yerr=ci, capsize=3, color=COL[arm], alpha=0.8)
            ax.scatter([j] * len(g), g.dominance.values - base, color="0.3", s=14, zorder=3)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_title(f"alpha={alpha:g}: dominance shift vs baseline"); ax.set_ylabel("dominance(arm) - dominance(baseline)")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Forward vs gradient pathway: learned modality-dominance shift (A scaled, 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "disentangle_arms_contribution.png", dpi=150); plt.close(fig)

    # Fig 2: distributions (per-seed dominance) at alpha=100
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for j, arm in enumerate(ARMS):
        g = ok[(ok.alpha == 100.0) & (ok.arm == arm)]
        ax.scatter([j] * len(g), g.dominance.values, color=COL[arm], s=40)
        ax.plot([j - 0.25, j + 0.25], [g.dominance.mean()] * 2, color=COL[arm], lw=2)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS); ax.set_ylabel("dominance Acc_A-Acc_V")
    ax.set_title("alpha=100: per-seed dominance distribution by arm"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "disentangle_arms_distribution.png", dpi=150); plt.close(fig)

    # Fig 3: trajectories (alpha=100): rho_fusion, rho_enc, grad_ratio, val_dominance
    tn = tj[(tj.alpha == 100.0)]
    fig, axx = plt.subplots(1, 4, figsize=(18, 4.2))
    keys = ["rho_fusion", "rho_enc", "grad_ratio", "val_dominance"]
    for k, key in enumerate(keys):
        for arm in ARMS:
            g = tn[tn.arm == arm]
            if len(g):
                m = g.groupby("epoch")[key].mean()
                axx[k].plot(m.index, m.values, color=COL[arm], label=arm, lw=1.7)
        axx[k].set_title(key); axx[k].set_xlabel("epoch"); axx[k].grid(alpha=0.25); axx[k].legend(fontsize=7)
        if key in ("rho_fusion", "rho_enc", "grad_ratio"):
            axx[k].set_yscale("log"); axx[k].axhline(1, color="0.6", ls=":")
    fig.suptitle("alpha=100 training trajectories by arm (mean of 5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "disentangle_trajectories.png", dpi=150); plt.close(fig)

    # Fig 4: compensation — rho_enc (does encoder counteract) by arm and alpha
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for arm in ARMS:
        xs, ys, es = [], [], []
        for alpha in ALPHAS:
            g = ok[(ok.alpha == alpha) & (ok.arm == arm)]
            m, sd, ci = _ci(g.rho_enc.values)
            xs.append(math.log10(alpha) if alpha > 0 else 0); ys.append(m); es.append(ci)
        ax.errorbar(xs, ys, yerr=es, marker="o", color=COL[arm], label=arm, capsize=3)
    ax.axhline(1, color="0.6", ls=":"); ax.set_xlabel("log10 alpha"); ax.set_ylabel(r"$\rho_{enc}=\|h_A\|/\|h_V\|$ (pre-arm)")
    ax.set_title("Encoder response (compensation) by arm"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "disentangle_compensation.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
