"""Phase 2 #2: per-modality heads (logit-sum fusion) vs shared head.
Prop 1's shared-head caveat: under forward scaling the shared head's gradient
along A also carries alpha (Adam does not fully cancel it). Separate per-modality
heads with logit-sum fusion remove that cross-term. Test whether the SGD
forward != gradient probe_A difference (shared: -0.038 vs +0.015) shrinks with
separate heads. Adam + SGD, baseline/forward/gradient, alpha=100, 10 seeds.
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
from experiments.train import _to_device
from experiments.disentangle_scale_grad import arm_apply, _reps
from models.multimodal_model import build_model
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed
from analysis.probes import linear_probe_accuracy

ALPHA = 100.0


def sep_logits(m, ha_head, hb_head, x_a, x_v, alpha, arm, mask_a=1.0, mask_v=1.0):
    h_a, h_b = _reps(m, x_a, x_v)
    h_a = arm_apply(h_a, alpha, arm)
    return ha_head(mask_a * h_a) + hb_head(mask_v * h_b)      # logit-sum fusion, separate heads


def train(cfg, ds, device, seed, arm, optimizer):
    set_seed(seed, True)
    data = _to_device(ds, device); nc = ds.meta["n_classes"]; rep = cfg["model"]["rep_dim"]
    a_tr, v_tr, y_tr = data["train"]; a_te, v_te, y_te = data["test"]
    m = build_model(deepcopy(cfg), ds.meta["n_dim_a"], ds.meta["n_dim_v"], nc).to(device)
    ha_head = nn.Linear(rep, nc).to(device); hb_head = nn.Linear(rep, nc).to(device)
    params = list(m.enc_a.parameters()) + list(m.enc_v.parameters()) + \
        list(m.norm_a.parameters()) + list(m.norm_v.parameters()) + \
        list(ha_head.parameters()) + list(hb_head.parameters())
    lr = float(cfg["train"]["lr"])
    opt = torch.optim.Adam(params, lr=lr) if optimizer == "adam" else torch.optim.SGD(params, lr=lr, momentum=0.9)
    lf = nn.CrossEntropyLoss()
    epochs = int(cfg["train"]["epochs"]); bs = int(cfg["train"]["batch_size"]); n = y_tr.numel()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(sep_logits(m, ha_head, hb_head, a_tr[idx], v_tr[idx], ALPHA, arm), y_tr[idx])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        ha_tr, hb_tr = _reps(m, a_tr, v_tr); ha_te, hb_te = _reps(m, a_te, v_te)
    probe_a = linear_probe_accuracy(ha_tr.detach(), y_tr, ha_te.detach(), y_te, nc, device)
    probe_b = linear_probe_accuracy(hb_tr.detach(), y_tr, hb_te.detach(), y_te, nc, device)
    with torch.no_grad():
        acc_a = float((sep_logits(m, ha_head, hb_head, a_te, v_te, ALPHA, arm, 1.0, 0.0).argmax(1) == y_te).float().mean())
        acc_v = float((sep_logits(m, ha_head, hb_head, a_te, v_te, ALPHA, arm, 0.0, 1.0).argmax(1) == y_te).float().mean())
    return {"seed": seed, "arm": arm, "optimizer": optimizer, "probe_a": probe_a, "probe_b": probe_b,
            "dominance": acc_a - acc_v}


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()
    for opt in ["adam", "sgd"]:
        for arm in ["baseline", "forward", "gradient", "ordinary"]:
            for seed in range(10):
                p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0)
                rows.append(train(cfg, generate_dataset(p), device, seed, arm, opt))
    df = pd.DataFrame(rows); df.to_csv(out / "phase2_heads.csv", index=False)

    def pci(opt, arm, col):
        b = df[(df.optimizer == opt) & (df.arm == "baseline")].sort_values("seed")[col].values
        x = df[(df.optimizer == opt) & (df.arm == arm)].sort_values("seed")[col].values
        dd = x - b; sd = dd.std(ddof=1); return dd.mean(), 1.96 * sd / math.sqrt(len(dd))
    print("\n=== SEPARATE per-modality heads (logit-sum): probe_A / dominance shift vs baseline ===")
    for opt in ["adam", "sgd"]:
        for arm in ["forward", "gradient", "ordinary"]:
            pa, pc = pci(opt, arm, "probe_a"); da, dc = pci(opt, arm, "dominance")
            print(f"  {opt} {arm:9s}: probe_A={pa:+.4f}±{pc:.4f}  dom={da:+.4f}±{dc:.4f}")
    json.dump({"git_commit": git_hash(), "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_heads_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
