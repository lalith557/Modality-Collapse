"""Phase 2 P3 (Adam eps sweep) and P4 (SGD eta/alpha exact-cancellation).

P3: Adam gradient-only at alpha=100, eps in {1e-8,1e-6,1e-4,1e-2}. Prop 1 says
    the s_g cancellation needs eps -> 0; once eps ~ RMS(grad), the gradient-only
    effect should RETURN (dominance shifts, update ratio rises).
P4: SGD gradient-only at alpha=100 with encoder-A LR = eta/alpha. Prop 1:
    m_SGD = s_f * s_g * (1/alpha) = 1 -> must reproduce BASELINE.

10 seeds, synthetic. Reuses phase2_mechanism.eval_all and the operators.

Usage: python experiments/phase2_eps_cancel.py --seeds 10
"""
from __future__ import annotations
import argparse, json, sys, time
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from data.synthetic import generate_dataset
from experiments.train import _to_device
from experiments.disentangle_scale_grad import arm_forward
from experiments.phase2_mechanism import eval_all
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ALPHA = 100.0


def _enc_groups(model, lr, lr_a=None):
    lr_a = lr if lr_a is None else lr_a
    ea = {id(p) for p in model.enc_a.parameters()}
    rest = [p for p in model.parameters() if id(p) not in ea]
    return [{"params": list(model.enc_a.parameters()), "lr": lr_a}, {"params": rest, "lr": lr}]


def train(cfg, ds, device, seed, arm, optimizer, eps=1e-8, lr_a_mult=1.0):
    set_seed(seed, True)
    data = _to_device(ds, device); nc = ds.meta["n_classes"]
    a_tr, v_tr, y_tr = data["train"]
    model = build_model(cfg, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    lr = float(cfg["train"]["lr"])
    groups = _enc_groups(model, lr, lr * lr_a_mult)
    if optimizer == "adam":
        opt = torch.optim.Adam(groups, lr=lr, eps=eps, weight_decay=0.0)
    else:
        opt = torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=0.0)
    lf = nn.CrossEntropyLoss()
    epochs = int(cfg["train"]["epochs"]); bs = int(cfg["train"]["batch_size"]); n = y_tr.numel()
    ea = list(model.enc_a.parameters()); ev = list(model.enc_v.parameters())
    ur = float("nan"); diverged = False
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); ua = ub = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            logits, *_ = arm_forward(model, a_tr[idx], v_tr[idx], ALPHA, arm)
            loss = lf(logits, y_tr[idx])
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward()
            pa = [p.detach().clone() for p in ea]; pv = [p.detach().clone() for p in ev]
            opt.step()
            ua += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ea, pa)) ** 0.5
            ub += sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(ev, pv)) ** 0.5
        if diverged:
            break
        ur = (ua / ub) if ub > 0 else float("nan")
    if diverged:
        return None
    m = eval_all(model, data, 1.0, nc, device)  # gradient-only: forward unscaled at eval
    return {"update_ratio": ur, **m}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds)); rows = []; t0 = time.time()

    def dss(seed):
        p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0); return generate_dataset(p)

    print("=== P3: Adam gradient-only, eps sweep (alpha=100) ===")
    for eps in [1e-8, 1e-6, 1e-4, 1e-2]:
        for seed in seeds:
            r = train(cfg, dss(seed), device, seed, "gradient", "adam", eps=eps)
            if r: rows.append({"test": "P3_eps", "eps": eps, "seed": seed, **r})
        g = [r for r in rows if r.get("test") == "P3_eps" and r["eps"] == eps]
        print(f"  eps={eps:.0e}: dom={np.mean([r['dominance'] for r in g]):+.4f} "
              f"upd_ratio={np.mean([r['update_ratio'] for r in g]):.2f} probe_a={np.mean([r['probe_a'] for r in g]):.3f}")

    print("=== P4: SGD gradient-only with enc-A LR = eta/alpha (should == baseline) ===")
    for seed in seeds:
        rb = train(cfg, dss(seed), device, seed, "baseline", "sgd")
        rg = train(cfg, dss(seed), device, seed, "gradient", "sgd")                      # normal gradient-only
        rc = train(cfg, dss(seed), device, seed, "gradient", "sgd", lr_a_mult=1.0 / ALPHA)  # cancellation
        for lab, r in [("baseline", rb), ("gradient", rg), ("gradient_lr_over_alpha", rc)]:
            if r: rows.append({"test": "P4_cancel", "variant": lab, "seed": seed, **r})
    for lab in ["baseline", "gradient", "gradient_lr_over_alpha"]:
        g = [r for r in rows if r.get("test") == "P4_cancel" and r["variant"] == lab]
        print(f"  {lab:24s}: dom={np.mean([r['dominance'] for r in g]):+.4f} "
              f"upd_ratio={np.mean([r['update_ratio'] for r in g]):.2f}")

    pd.DataFrame(rows).to_csv(out / "phase2_eps_cancel.csv", index=False)
    json.dump({"git_commit": git_hash(), "seeds": seeds, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_eps_cancel_config.json", "w"), indent=2, default=str)
    print(f"\nWALL {time.time()-t0:.1f}s -> {out}/phase2_eps_cancel.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
