"""Phase 2 #3 + #1: plain encoder-A LR sweep (NO scaling) per optimizer, as the
reference curve for 'A's probe vs effective multiplier m'. Logs final train loss
(instability check for large LR). The scaling arms (forward/gradient/ordinary)
are overlaid at their predicted m from phase2_primary_seeds10.csv in analysis.
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

RMULT = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]   # enc-A LR multiplier


def train(cfg, ds, device, seed, r, optimizer):
    set_seed(seed, True)
    data = _to_device(ds, device); nc = ds.meta["n_classes"]
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]
    cfg2 = deepcopy(cfg); cfg2["train"]["optimizer"] = optimizer
    cfg2["train"]["lr_a"] = float(cfg2["train"]["lr"]) * r        # enc-A LR x r, NO scaling
    model = build_model(cfg2, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    opt = _optimizer(model, cfg2, device); lf = nn.CrossEntropyLoss()
    epochs = int(cfg2["train"]["epochs"]); bs = int(cfg2["train"]["batch_size"]); n = y_tr.numel()
    last_loss = float("nan"); diverged = False
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device); el = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_forward(model, a_tr[idx], v_tr[idx], 1.0, "baseline")[0], y_tr[idx])
            if not torch.isfinite(loss):
                diverged = True; break
            loss.backward(); opt.step(); el += float(loss.detach()) * len(idx)
        if diverged:
            break
        last_loss = el / n
    if diverged:
        return {"seed": seed, "r": r, "optimizer": optimizer, "diverged": True}
    va = arm_contrib(model, a_va, v_va, y_va, 1.0, "baseline", nc)["acc_av"]
    m = eval_all(model, data, 1.0, nc, device)
    return {"seed": seed, "r": r, "optimizer": optimizer, "diverged": False,
            "train_loss": last_loss, "val_acc": va, "probe_a": m["probe_a"], "probe_b": m["probe_b"],
            "dominance": m["dominance"], "acc_a": m["acc_a"], "acc_v": m["acc_v"]}


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()
    for opt in ["adam", "sgd"]:
        for r in RMULT:
            for seed in range(10):
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
                rows.append(train(cfg, generate_dataset(p), device, seed, r, opt))
            g = [x for x in rows if x["optimizer"] == opt and x["r"] == r and not x.get("diverged")]
            print(f"  {opt} r={r:5g}: probe_a={np.mean([x['probe_a'] for x in g]):.3f} "
                  f"train_loss={np.mean([x['train_loss'] for x in g]):.3f} "
                  f"val_acc={np.mean([x['val_acc'] for x in g]):.3f} div={10-len(g)}")
    pd.DataFrame(rows).to_csv(out / "phase2_lrsweep.csv", index=False)
    json.dump({"git_commit": git_hash(), "rmult": RMULT, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_lrsweep_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s -> {out}/phase2_lrsweep.csv")


if __name__ == "__main__":
    main()
