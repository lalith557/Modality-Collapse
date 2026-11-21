"""Phase 2 condition #3: baseline-corrected c(alpha) curve for synthetic (forward
arm), both optimizers. Logs rho_enc, rho_fus; c = 1 - ln(rho_fus(a)/rho_fus(1))/ln a.
"""
from __future__ import annotations
import json, math, sys, time
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from data.synthetic import generate_dataset
from experiments.train import _optimizer, _to_device
from experiments.disentangle_scale_grad import arm_forward, arm_rho
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def train_rho(cfg, ds, device, seed, alpha, arm, optimizer):
    set_seed(seed, True)
    data = _to_device(ds, device)
    a_tr, v_tr, y_tr = data["train"]; a_te, v_te, y_te = data["test"]
    cfg2 = deepcopy(cfg); cfg2["train"]["optimizer"] = optimizer
    model = build_model(cfg2, ds.meta["n_dim_a"], ds.meta["n_dim_v"], ds.meta["n_classes"]).to(device)
    opt = _optimizer(model, cfg2, device); lf = nn.CrossEntropyLoss()
    epochs = int(cfg2["train"]["epochs"]); bs = int(cfg2["train"]["batch_size"]); n = y_tr.numel()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_forward(model, a_tr[idx], v_tr[idx], alpha, arm)[0], y_tr[idx])
            loss.backward(); opt.step()
    re, rf = arm_rho(model, a_te, v_te, alpha, arm)
    return re, rf


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()
    for opt in ["adam", "sgd"]:
        rf_base = {}
        for alpha in ALPHAS:
            res = []
            for seed in range(10):
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
                arm = "baseline" if alpha == 1.0 else "forward"
                re, rf = train_rho(cfg, generate_dataset(p), device, seed, alpha, arm, opt)
                res.append((re, rf))
            rf_mean = float(np.mean([r[1] for r in res]))
            if alpha == 1.0:
                rf_base[opt] = rf_mean
            c = (1 - math.log(rf_mean / rf_base[opt]) / math.log(alpha)) if alpha != 1.0 else float("nan")
            rows.append({"optimizer": opt, "alpha": alpha, "rho_fus": rf_mean,
                         "rho_fus_base": rf_base[opt], "c_corrected": c})
            print(f"  {opt} a={alpha:5g}: rho_fus={rf_mean:.3f} c={c:.3f}")
    pd.DataFrame(rows).to_csv(out / "phase2_ccurve.csv", index=False)
    json.dump({"git_commit": git_hash(), "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_ccurve_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s -> {out}/phase2_ccurve.csv")


if __name__ == "__main__":
    main()
