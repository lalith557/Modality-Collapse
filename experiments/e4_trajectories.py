"""E4 representative training trajectories (transient vs persistent).

Re-runs a handful of representative (scale, lr, norm) cells with the SAME config,
generator and seeds as the primary grid, capturing per-epoch weak-contribution and
log rho, to distinguish transient imbalance from persistent collapse.
Does NOT touch the primary grid outputs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.synthetic import generate_dataset
from experiments.train import train_model
from msc.device import get_device
from msc.runlog import load_config

CONDS = [
    ("balanced (1,1)", 1.0, 1.0, "none", "#000000"),
    ("pure scale (20,1)", 20.0, 1.0, "none", "#1f77b4"),
    ("pure LR (1,20)", 1.0, 20.0, "none", "#2ca02c"),
    ("aligned (20,20)", 20.0, 20.0, "none", "#d62728"),
    ("opposed (20,1/20)", 20.0, 0.05, "none", "#ff7f0e"),
    ("aligned+zscore (20,20)", 20.0, 20.0, "zscore", "#9467bd"),
]


def run(cfg, device, s, l, norm, seeds):
    base = float(cfg["train"]["lr"])
    weak_t, logrho_t, gain_t = [], [], []
    for seed in seeds:
        p = dict(cfg["data"]); p.update(data_seed=seed, alpha_a=s ** 0.5, alpha_v=s ** -0.5)
        ds = generate_dataset(p)
        cfg["model"]["normalization"] = norm
        cfg["train"]["lr_a"] = base * (l ** 0.5); cfg["train"]["lr_v"] = base * (l ** -0.5)
        res = train_model(cfg, ds, device, seed=seed)
        h = sorted(res["rho_history"], key=lambda r: r["epoch"])
        weak_t.append([min(r["val_delta_a"], r["val_delta_v"]) for r in h])
        logrho_t.append([math.log(r["rho"]) for r in h])
        gain_t.append([r["val_gain"] for r in h])
    eps = [r["epoch"] for r in h]
    return eps, np.mean(weak_t, 0), np.mean(logrho_t, 0), np.mean(gain_t, 0)


def main():
    cfg = load_config(ROOT / "configs" / "boundary.yaml")
    device = get_device(cfg["device"])
    seeds = cfg["seed_policy"]["seeds"]
    baseline = None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for label, s, l, norm, color in CONDS:
        eps, weak, logrho, gain = run(cfg, device, s, l, norm, seeds)
        if label.startswith("balanced"):
            baseline = weak[-5:].mean()
        ls = "--" if norm == "zscore" else "-"
        axes[0].plot(eps, weak, ls, color=color, label=label, lw=1.8)
        axes[1].plot(eps, logrho, ls, color=color, label=label, lw=1.8)
    axes[0].axhline(0.5 * baseline, color="0.5", ls=":", lw=1, label="collapse threshold (0.5×baseline)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("weak-modality contribution min(Δ_A,Δ_V)")
    axes[0].set_title("Weak-modality contribution over training"); axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel(r"$\log \rho$ (fusion-boundary)")
    axes[1].set_title("Activation-norm ratio over training"); axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("E4 representative trajectories — transient imbalance vs persistent collapse (mean of 5 seeds)")
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "e4_final_trajectories.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
