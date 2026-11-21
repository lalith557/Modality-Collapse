"""Phase 2 (CVPR revision): mechanism + theory tests on synthetic.

Arms:
  baseline, forward, gradient, ordinary        (standard; reuse arm_apply)
  lrmatch   : NO forward/grad scaling; encoder-A Adam LR x alpha (predicted
              forward multiplier). Tests H-lr: is the forward effect just an
              effective-LR effect? (Adam only.)
  forward_k : forward scaling applied only for the first k epochs, then removed.
              Tests H-init: initial-logit-imbalance vs sustained-LR effect.

Logs probe / mean-ablation / Shapley at EVERY alpha. 10 seeds, paired across
seeds. Probe trained on TRAIN split only, identical hyperparams for every arm.
Prior experiments are untouched.

Usage:
  python experiments/phase2_mechanism.py --pilot
  python experiments/phase2_mechanism.py --seeds 10 --alphas 1,3,10,30,100,300
"""
from __future__ import annotations
import argparse, json, math, sys, time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch, torch.nn as nn
from data.synthetic import generate_dataset
from experiments.train import _optimizer, _to_device
from experiments.disentangle_scale_grad import ForwardScale, _reps, arm_forward
from models.fusion import fuse
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed
from analysis.probes import linear_probe_accuracy

NAN = float("nan")
CORE = ["baseline", "forward", "gradient", "ordinary"]
EXTRA = ["lrmatch", "forward_k"]          # Adam-only mechanism arms
K_EPOCHS = 5                              # forward_k: scale on for first 5 epochs


def eval_all(model, data, fa_scale, nc, device, with_probe=True):
    """Decoupled metrics. fa_scale multiplies A's reps at the fusion input."""
    a_tr, v_tr, y_tr = data["train"]; a_te, v_te, y_te = data["test"]
    model.eval()
    with torch.no_grad():
        ha_tr, hv_tr = _reps(model, a_tr, v_tr); ha_te, hv_te = _reps(model, a_te, v_te)
    ha_tr, hv_tr, ha_te, hv_te = (t.detach() for t in (ha_tr, hv_tr, ha_te, hv_te))
    probe_a = linear_probe_accuracy(ha_tr, y_tr, ha_te, y_te, nc, device) if with_probe else NAN
    probe_b = linear_probe_accuracy(hv_tr, y_tr, hv_te, y_te, nc, device) if with_probe else NAN
    with torch.no_grad():
        fa_te = fa_scale * ha_te; fv_te = hv_te
        mfa = (fa_scale * ha_tr).mean(0, keepdim=True).expand_as(fa_te)
        mfv = hv_tr.mean(0, keepdim=True).expand_as(fv_te)

        def acc(fa, fv):
            return float((model.head(fuse(model.fusion, fa, fv)).argmax(1) == y_te).float().mean())
        v_both = acc(fa_te, fv_te); v_A = acc(fa_te, mfv); v_B = acc(mfa, fv_te); v_empty = acc(mfa, mfv)
        z_A = acc(fa_te, torch.zeros_like(fv_te)); z_B = acc(torch.zeros_like(fa_te), fv_te)
    return {"acc_a": z_A, "acc_v": z_B, "dominance": z_A - z_B,
            "meanabl_dominance": v_A - v_B, "shapley_a": 0.5 * (v_A - v_empty) + 0.5 * (v_both - v_B),
            "shapley_b": 0.5 * (v_B - v_empty) + 0.5 * (v_both - v_A),
            "probe_a": probe_a, "probe_b": probe_b, "v_both": v_both}


def train_one(cfg, ds, device, seed, alpha, arm, optimizer):
    set_seed(seed, True)
    data = _to_device(ds, device); nc = ds.meta["n_classes"]
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]
    cfg2 = deepcopy(cfg); cfg2["train"]["optimizer"] = optimizer
    if arm == "lrmatch":
        cfg2["train"]["lr_a"] = float(cfg2["train"]["lr"]) * alpha     # enc-A LR x predicted multiplier
    model = build_model(cfg2, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    opt = _optimizer(model, cfg2, device)
    lf = nn.CrossEntropyLoss()
    epochs = int(cfg2["train"]["epochs"]); bs = int(cfg2["train"]["batch_size"]); n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    diverged = False; gr = ur = NAN
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); ga = gb = ua = ub = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            # per-step forward transform
            if arm in ("baseline", "lrmatch"):
                step_arm = "baseline"
            elif arm == "forward_k":
                step_arm = "forward" if ep <= K_EPOCHS else "baseline"
            else:
                step_arm = arm
            logits, *_ = arm_forward(model, a_tr[idx], v_tr[idx], alpha, step_arm)
            loss = lf(logits, y_tr[idx])
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            ga += sum(float(p.grad.pow(2).sum()) for p in ea) ** 0.5
            gb += sum(float(p.grad.pow(2).sum()) for p in ev) ** 0.5
            pa = [p.detach().clone() for p in ea]; pv = [p.detach().clone() for p in ev]
            opt.step()
            ua += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ea, pa)) ** 0.5
            ub += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ev, pv)) ** 0.5
        if diverged:
            break
        gr = (ga / gb) if gb > 0 else NAN; ur = (ua / ub) if ub > 0 else NAN
    if diverged:
        return {"seed": seed, "alpha": alpha, "arm": arm, "optimizer": optimizer, "diverged": True}
    fa_scale = alpha if arm in ("forward", "ordinary") else 1.0    # deployed scale
    m = eval_all(model, data, fa_scale, nc, device)
    return {"seed": seed, "alpha": alpha, "arm": arm, "optimizer": optimizer, "diverged": False,
            "grad_ratio_final": gr, "update_ratio_final": ur, **m}


def run(cfg, device, out, seeds, alphas):
    finals = []; t0 = time.time()
    plan = [(o, "core", a) for o in ("adam", "sgd") for a in CORE] + [("adam", "extra", a) for a in EXTRA]
    for opt_name, _grp, arm in plan:
        for alpha in alphas:
            if arm != "baseline" and alpha == 1.0:
                continue  # all arms == baseline at alpha=1; run baseline only
            for seed in seeds:
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
                ds = generate_dataset(p)
                finals.append(train_one(cfg, ds, device, seed, alpha, arm, opt_name))
            g = [f for f in finals if f["optimizer"] == opt_name and f["alpha"] == alpha and f["arm"] == arm and not f.get("diverged")]
            dom = np.mean([f["dominance"] for f in g]) if g else NAN
            pa = np.mean([f["probe_a"] for f in g]) if g else NAN
            nd = sum(1 for f in finals if f["optimizer"] == opt_name and f["alpha"] == alpha and f["arm"] == arm and f.get("diverged"))
            print(f"  {opt_name} a={alpha:5g} {arm:9s} dom={dom:+.3f} probe_a={pa:.3f} div={nd} ({time.time()-t0:.0f}s)")
    return finals, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alphas", type=str, default="1,3,10,30,100,300")
    args = ap.parse_args()
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; out.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        seeds = [0]; alphas = [100.0]
    else:
        seeds = list(range(args.seeds)); alphas = [float(x) for x in args.alphas.split(",")]
    finals, wall = run(cfg, device, out, seeds, alphas)
    tag = "pilot" if args.pilot else f"seeds{args.seeds}"
    pd.DataFrame(finals).to_csv(out / f"phase2_primary_{tag}.csv", index=False)
    json.dump({"git_commit": git_hash(), "seeds": seeds, "alphas": alphas, "k_epochs": K_EPOCHS,
               "n_runs": len(finals), "wall_s": round(wall, 1)}, open(out / f"phase2_config_{tag}.json", "w"), indent=2, default=str)
    print(f"\nWALL {wall:.1f}s / {len(finals)} runs -> {out}/phase2_primary_{tag}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
