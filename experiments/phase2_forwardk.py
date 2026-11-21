"""Phase 2 condition #1: forward_k at SEVERAL k (a null at too-short k is
uninformative). Adam, alpha=100, 10 seeds. k=40 (= all epochs) must recover
forward. Reuses phase2_mechanism.train_one via K_EPOCHS override.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import experiments.phase2_mechanism as pm
from data.synthetic import generate_dataset
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config

ALPHA = 100.0
KS = [1, 5, 10, 20, 40]


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()

    def ds(seed):
        p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0); return generate_dataset(p)

    # reference: baseline and forward (full scaling = k=40)
    for arm in ["baseline", "forward"]:
        for seed in range(10):
            r = pm.train_one(cfg, ds(seed), device, seed, ALPHA, arm, "adam")
            rows.append({"k": (40 if arm == "forward" else 0), "arm": arm, **r})
    # forward_k sweep
    for k in KS:
        pm.K_EPOCHS = k
        for seed in range(10):
            r = pm.train_one(cfg, ds(seed), device, seed, ALPHA, "forward_k", "adam")
            rows.append({"k": k, "arm": "forward_k", **r})
        g = [x for x in rows if x["arm"] == "forward_k" and x["k"] == k and not x.get("diverged")]
        print(f"  k={k:2d}: dom={np.mean([x['dominance'] for x in g]):+.4f} probe_a={np.mean([x['probe_a'] for x in g]):.4f}")
    df = pd.DataFrame(rows); df.to_csv(out / "phase2_forwardk.csv", index=False)
    bl = df[df.arm == "baseline"]; fw = df[df.arm == "forward"]
    print(f"  [ref] baseline dom={bl.dominance.mean():+.4f} probe={bl.probe_a.mean():.4f} | "
          f"forward dom={fw.dominance.mean():+.4f} probe={fw.probe_a.mean():.4f}")
    json.dump({"git_commit": git_hash(), "ks": KS, "alpha": ALPHA, "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_forwardk_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s -> {out}/phase2_forwardk.csv")


if __name__ == "__main__":
    main()
