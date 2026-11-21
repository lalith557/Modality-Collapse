"""Phase 2 #4: tune the GLOBAL base LR per optimizer on the validation split,
then rerun baseline / forward / gradient at the tuned LR. Tests whether SGD's
'improves at small alpha' is just an undertrained baseline.
"""
from __future__ import annotations
import json, sys, time
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from data.synthetic import generate_dataset
from experiments.train import _optimizer, _to_device
from experiments.disentangle_scale_grad import arm_forward, arm_contrib
from experiments.phase2_mechanism import eval_all
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

GRID = {"adam": [3e-4, 1e-3, 3e-3, 1e-2], "sgd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]}
ALPHA = 100.0


def train(cfg, ds, device, seed, lr, optimizer, arm, alpha=1.0):
    set_seed(seed, True)
    data = _to_device(ds, device); nc = ds.meta["n_classes"]
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]
    cfg2 = deepcopy(cfg); cfg2["train"]["optimizer"] = optimizer; cfg2["train"]["lr"] = lr
    model = build_model(cfg2, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    opt = _optimizer(model, cfg2, device); lf = nn.CrossEntropyLoss()
    epochs = int(cfg2["train"]["epochs"]); bs = int(cfg2["train"]["batch_size"]); n = y_tr.numel()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_forward(model, a_tr[idx], v_tr[idx], alpha, arm)[0], y_tr[idx])
            if not torch.isfinite(loss):
                return None
            loss.backward(); opt.step()
    va = arm_contrib(model, a_va, v_va, y_va, alpha, arm, nc)["acc_av"]
    fa_scale = alpha if arm in ("forward", "ordinary") else 1.0
    m = eval_all(model, data, fa_scale, nc, device)
    return {"val_acc": va, **m}


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; best = {}; t0 = time.time()

    def ds(seed):
        p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0); return generate_dataset(p)

    # 1) tune base LR on val (baseline arm, 5 seeds)
    print("=== LR grid (val_acc, baseline) ===")
    for opt in ["adam", "sgd"]:
        scores = {}
        for lr in GRID[opt]:
            vs = [train(cfg, ds(s), device, s, lr, opt, "baseline") for s in range(5)]
            vs = [v for v in vs if v]
            scores[lr] = float(np.mean([v["val_acc"] for v in vs])) if vs else -1
            print(f"  {opt} lr={lr:.0e}: val_acc={scores[lr]:.4f}")
        best[opt] = max(scores, key=scores.get)
        print(f"  -> best {opt} base lr = {best[opt]:.0e}")

    # 2) rerun key arms at tuned base LR (10 seeds, alpha=100)
    print("=== key comparisons at tuned base LR ===")
    for opt in ["adam", "sgd"]:
        for arm in ["baseline", "forward", "gradient"]:
            for seed in range(10):
                r = train(cfg, ds(seed), device, seed, best[opt], opt, arm, ALPHA)
                if r:
                    rows.append({"optimizer": opt, "tuned_lr": best[opt], "arm": arm, "seed": seed, **r})
    df = pd.DataFrame(rows); df.to_csv(out / "phase2_tunedbase.csv", index=False)
    import math
    for opt in ["adam", "sgd"]:
        b = df[(df.optimizer == opt) & (df.arm == "baseline")].sort_values("seed")
        print(f"\n--- {opt} (tuned base lr={best[opt]:.0e}) ---  baseline probe_a={b.probe_a.mean():.3f}")
        for arm in ["forward", "gradient"]:
            a = df[(df.optimizer == opt) & (df.arm == arm)].sort_values("seed")
            dd = a.probe_a.values - b.probe_a.values; sd = dd.std(ddof=1)
            ddom = a.dominance.values - b.dominance.values
            print(f"    {arm:9s}: probe_a shift={dd.mean():+.4f}±{1.96*sd/math.sqrt(len(dd)):.4f}  "
                  f"dom shift={ddom.mean():+.4f}")
    json.dump({"git_commit": git_hash(), "best_lr": best, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_tunedbase_config.json", "w"), indent=2, default=str)
    print(f"\nWALL {time.time()-t0:.1f}s -> {out}/phase2_tunedbase.csv")


if __name__ == "__main__":
    main()
