"""Phase 3 (scoped): CREMA-D, my actor-disjoint split, decoupled metrics, tuned LR,
AdamW + SGD, baseline/forward/gradient, alpha=100, 5 seeds.
Tests whether W2 (masked-accuracy-only effect) holds under a tuned baseline and SGD.
AdamW baseline/forward reuse phase2_cremad_probe.csv (weight_decay=0 => AdamW=Adam).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from experiments.cremad_run import _chunks, load_features
from experiments.cremad_disentangle import arm_logits
from experiments.phase2_cremad_probe import decoupled, ALPHA
from models.cremad_model import CremadMultimodalModel
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

ARMS = ["baseline", "forward", "gradient"]
SGD_GRID = [3e-3, 1e-2, 3e-2]


@torch.no_grad()
def val_acc(model, feats, arm):
    a, v, y = feats["val"]; model.eval(); preds = []
    for sl in _chunks(a.shape[0]):
        preds.append(arm_logits(model, a[sl], v[sl], ALPHA, arm).argmax(1))
    return float((torch.cat(preds) == y).float().mean())


def train(cfg, feats, device, seed, arm, optimizer, lr, with_dec=True):
    set_seed(seed, True); nc = cfg["data"]["n_classes"]
    a_tr, v_tr, y_tr = feats["train"]
    model = CremadMultimodalModel(nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                                  normalization="none", head=cfg["model"]["head"]).to(device)
    opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0) if optimizer == "adamw"
           else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))
    lf = nn.CrossEntropyLoss(); epochs = cfg["train"]["epochs"]; bs = cfg["train"]["batch_size"]; n = y_tr.numel()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lf(arm_logits(model, a_tr[idx], v_tr[idx], ALPHA, arm), y_tr[idx])
            if not torch.isfinite(loss):
                return {"seed": seed, "arm": arm, "optimizer": optimizer, "lr": lr, "diverged": True, "val_acc": 0.0}
            loss.backward(); opt.step()
    va = val_acc(model, feats, arm)
    out = {"seed": seed, "arm": arm, "optimizer": optimizer, "lr": lr, "diverged": False, "val_acc": va}
    if with_dec:
        out.update(decoupled(model, feats, nc, device, arm))
    return out


def main():
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = load_features(device); out = ROOT / "results" / "phase3_cremad"; out.mkdir(parents=True, exist_ok=True)
    rows = []; t0 = time.time()

    # 1) tune SGD base lr on val (baseline, 2 seeds)
    print("=== SGD lr tuning (val_acc, baseline) ===")
    best_lr, best_va = SGD_GRID[0], -1
    for lr in SGD_GRID:
        vas = [train(cfg, feats, device, s, "baseline", "sgd", lr, with_dec=False)["val_acc"] for s in range(2)]
        mva = float(np.mean(vas)); print(f"  sgd lr={lr:.0e}: val_acc={mva:.4f} ({time.time()-t0:.0f}s)")
        if mva > best_va:
            best_va, best_lr = mva, lr
    print(f"  -> best SGD lr = {best_lr:.0e}")

    # 2) AdamW gradient arm (baseline/forward reused from phase2_cremad_probe.csv)
    print("=== AdamW gradient arm (lr=1e-3, 5 seeds) ===")
    for seed in range(5):
        rows.append(train(cfg, feats, device, seed, "gradient", "adamw", 1e-3))
        print(f"  adamw gradient seed={seed}: dom={rows[-1].get('dominance',float('nan')):+.3f} ({time.time()-t0:.0f}s)")

    # 3) SGD baseline/forward at tuned lr (5 seeds); gradient 2 seeds to document stability
    print(f"=== SGD arms at lr={best_lr:.0e} ===")
    for arm, nseed in [("baseline", 5), ("forward", 5), ("gradient", 2)]:
        for seed in range(nseed):
            rows.append(train(cfg, feats, device, seed, arm, "sgd", best_lr))
            print(f"  sgd {arm:9s} seed={seed}: diverged={rows[-1]['diverged']} "
                  f"dom={rows[-1].get('dominance',float('nan')):+.3f} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows); df.to_csv(out / "phase3_cremad_scoped.csv", index=False)
    # merge existing AdamW baseline/forward
    ex = pd.read_csv(ROOT / "results" / "phase2_mechanism" / "phase2_cremad_probe.csv")
    ex = ex[ex.arm.isin(["baseline", "forward"])].copy(); ex["optimizer"] = "adamw"; ex["lr"] = 1e-3
    allp = pd.concat([ex, df], ignore_index=True); allp = allp[~allp.get("diverged", False).astype(bool)]

    def pci(opt, arm, col):
        b = allp[(allp.optimizer == opt) & (allp.arm == "baseline")].sort_values("seed")[col].values
        x = allp[(allp.optimizer == opt) & (allp.arm == arm)].sort_values("seed")[col].values
        k = min(len(b), len(x))
        if k < 2:
            return None
        dd = x[:k] - b[:k]; sd = dd.std(ddof=1)
        return dd.mean(), 1.96 * sd / math.sqrt(len(dd)), k
    # divergence report
    div = df[df.diverged.astype(bool)].groupby(["optimizer", "arm"]).size()
    print("\n=== divergence counts ===\n" + (div.to_string() if len(div) else "  none"))
    print("\n=== CREMA-D scoped: baseline->arm shifts (paired CI). Does the effect survive decoupled metrics? ===")
    for opt in ["adamw", "sgd"]:
        print(f"-- {opt} (lr={'1e-3' if opt=='adamw' else f'{best_lr:.0e}'}) --")
        for arm in ["forward", "gradient"]:
            for col in ["dominance", "meanabl_dominance", "shapley_audio", "probe_audio"]:
                r = pci(opt, arm, col)
                print(f"   {arm:9s} {col:18s}: {'n<2 (diverged/absent)' if r is None else f'{r[0]:+.4f} ± {r[1]:.4f} (n={r[2]})'}")
    json.dump({"git_commit": git_hash(), "best_sgd_lr": best_lr, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase3_cremad_scoped_config.json", "w"), indent=2, default=str)
    print(f"\nWALL {time.time()-t0:.1f}s -> {out}/phase3_cremad_scoped.csv")


if __name__ == "__main__":
    main()
