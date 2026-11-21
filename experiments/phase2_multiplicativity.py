"""Phase 2 #3: test SGD multiplicativity in the UNSATURATED regime.
Prop 1: SGD ordinary(alpha) has m=alpha^2, forward(alpha^2) and gradient(alpha^2)
have m=alpha^2. So ordinary(a) should match forward(a^2) and gradient(a^2).
Uses small alpha where dominance is NOT saturated. 10 seeds, paired.
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

SMALL = [1.5, 2.0, 3.0, 5.0]


def main():
    device = get_device({"type": "cuda", "allow_cpu": False}); print("Device:", device_report(device))
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    out = ROOT / "results" / "phase2_mechanism"; rows = []; t0 = time.time()

    def ds(seed):
        p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=1.0, alpha_v=1.0); return generate_dataset(p)

    for a in SMALL:
        combos = [("ordinary", a, a * a), ("forward", a * a, a * a), ("gradient", a * a, a * a)]
        for arm, alpha, m in combos:
            for seed in range(10):
                r = pm.train_one(cfg, ds(seed), device, seed, alpha, arm, "sgd")
                rows.append({"base_alpha": a, "arm": arm, "alpha": alpha, "pred_m": m, **r})
    df = pd.DataFrame(rows); df = df[~df.diverged.astype(bool)]
    df.to_csv(out / "phase2_multiplicativity.csv", index=False)
    print("\n=== SGD multiplicativity (unsaturated): ordinary(a) vs forward(a^2) vs gradient(a^2) ===")
    print(f"{'base a':>7} {'pred m':>7} {'ord dom':>18} {'fwd(a^2) dom':>18} {'grad(a^2) dom':>18}")
    for a in SMALL:
        def dom(arm, alpha):
            g = df[(df.arm == arm) & np.isclose(df.alpha, alpha)]
            return g.dominance.mean(), 1.96 * g.dominance.std(ddof=1) / math.sqrt(len(g))
        o = dom("ordinary", a); f = dom("forward", a * a); gr = dom("gradient", a * a)
        print(f"{a:7g} {a*a:7g} {o[0]:+.4f}±{o[1]:.4f}    {f[0]:+.4f}±{f[1]:.4f}    {gr[0]:+.4f}±{gr[1]:.4f}")
    json.dump({"git_commit": git_hash(), "wall_s": round(time.time() - t0, 1)},
              open(out / "phase2_multiplicativity_config.json", "w"), indent=2, default=str)
    print(f"WALL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
