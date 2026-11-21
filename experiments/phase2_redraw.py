"""Phase 2 #8: final-table robustness — REDRAW the full synthetic dataset per
seed (fresh class prototypes AND samples), so the CIs include data variation,
not just init/order. Key arms at alpha=100, both optimizers, 10 seeds.
"""
from __future__ import annotations
import json, math, sys, time
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


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()
    for opt in ["adam", "sgd"]:
        arms = ["baseline", "forward", "gradient", "ordinary"] + (["lrmatch"] if opt == "adam" else [])
        for arm in arms:
            for seed in range(10):
                p = dict(cfg["data"])
                p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0, prototype_seed=1000 + seed)  # fresh prototypes
                rows.append(pm.train_one(cfg, generate_dataset(p), device, seed, ALPHA, arm, opt))
    df = pd.DataFrame(rows); df = df[~df.diverged.astype(bool)]
    df.to_csv(out / "phase2_redraw.csv", index=False)

    def pci(opt, arm, col):
        b = df[(df.optimizer == opt) & (df.arm == "baseline")].sort_values("seed")[col].values
        x = df[(df.optimizer == opt) & (df.arm == arm)].sort_values("seed")[col].values
        dd = x - b; sd = dd.std(ddof=1); return dd.mean(), 1.96 * sd / math.sqrt(len(dd))
    print("\n=== REDRAWN prototypes per seed (alpha=100): dom & probe_A shifts vs baseline ===")
    for opt in ["adam", "sgd"]:
        arms = ["forward", "gradient", "ordinary"] + (["lrmatch"] if opt == "adam" else [])
        for arm in arms:
            dm, dc = pci(opt, arm, "dominance"); pm_, pc = pci(opt, arm, "probe_a")
            print(f"  {opt} {arm:9s}: dom={dm:+.4f}±{dc:.4f}  probe_A={pm_:+.4f}±{pc:.4f}")
    json.dump({"git_commit": git_hash(), "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_redraw_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
