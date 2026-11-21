"""Phase 2 #7: does A's probe degradation survive early stopping?
Track val acc per epoch, checkpoint the best-val model, probe at BEST-VAL and at
FINAL for Adam baseline vs forward (alpha=100). Log per-modality train/test
masked accuracy. 10 seeds.
"""
from __future__ import annotations
import copy, json, math, sys, time
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

ALPHA = 100.0


def run(cfg, ds, device, seed, arm, optimizer):
    set_seed(seed, True)
    data = _to_device(ds, device); nc = ds.meta["n_classes"]
    a_tr, v_tr, y_tr = data["train"]; a_va, v_va, y_va = data["val"]
    cfg2 = deepcopy(cfg); cfg2["train"]["optimizer"] = optimizer
    model = build_model(cfg2, ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    opt = _optimizer(model, cfg2, device); lf = nn.CrossEntropyLoss()
    epochs = int(cfg2["train"]["epochs"]); bs = int(cfg2["train"]["batch_size"]); n = y_tr.numel()
    best_va = -1; best_state = None; best_ep = -1
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_forward(model, a_tr[idx], v_tr[idx], ALPHA, arm)[0], y_tr[idx])
            loss.backward(); opt.step()
        va = arm_contrib(model, a_va, v_va, y_va, ALPHA, arm, nc)["acc_av"]
        if va > best_va:
            best_va = va; best_ep = ep; best_state = copy.deepcopy(model.state_dict())
    fa_scale = ALPHA if arm in ("forward", "ordinary") else 1.0
    m_final = eval_all(model, data, fa_scale, nc, device)
    model.load_state_dict(best_state)
    m_best = eval_all(model, data, fa_scale, nc, device)
    tr = arm_contrib(model, a_tr, v_tr, y_tr, ALPHA, arm, nc)
    return {"seed": seed, "arm": arm, "best_ep": best_ep,
            "probe_a_final": m_final["probe_a"], "probe_a_best": m_best["probe_a"],
            "dom_final": m_final["dominance"], "dom_best": m_best["dominance"],
            "train_acc_a": tr["acc_a"], "train_acc_v": tr["acc_v"]}


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()
    for arm in ["baseline", "forward"]:
        for seed in range(10):
            p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
            rows.append(run(cfg, generate_dataset(p), device, seed, arm, "adam"))
    df = pd.DataFrame(rows); df.to_csv(out / "phase2_earlystop.csv", index=False)
    b = df[df.arm == "baseline"].sort_values("seed"); f = df[df.arm == "forward"].sort_values("seed")
    def ci(x): x = np.asarray(x, float); return x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    print(f"\n  mean best-val epoch: baseline={b.best_ep.mean():.1f} forward={f.best_ep.mean():.1f} (of {cfg['train']['epochs']})")
    for tag, cb, cf in [("FINAL", "probe_a_final", "probe_a_final"), ("BEST-VAL", "probe_a_best", "probe_a_best")]:
        dd = f[cf].values - b[cb].values; m, c = ci(dd)
        print(f"  probe_A shift (forward-baseline) @ {tag:8}: {m:+.4f} ± {c:.4f}")
    json.dump({"git_commit": git_hash(), "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_earlystop_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
