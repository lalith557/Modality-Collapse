"""Publication-oriented plots for E2 (scale -> collapse)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _err(agg, key):
    x = [a["alpha_ratio"] for a in agg]
    mean = [a.get(f"{key}_mean") for a in agg]
    ci = [a.get(f"{key}_ci95", 0.0) for a in agg]
    return x, mean, ci


def _finish(ax):
    ax.set_xscale("log")
    ax.axvline(1.0, color="0.6", ls=":", lw=1, label="balanced (ratio=1)")
    ax.grid(True, which="both", alpha=0.25)


def plot_scale_vs_contribution(agg, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for key, lab, col in [("test_delta_a", r"$\Delta_A$ (A's contribution)", "#1f77b4"),
                          ("test_delta_v", r"$\Delta_V$ (V's contribution)", "#d62728")]:
        x, m, ci = _err(agg, key)
        ax.errorbar(x, m, yerr=ci, marker="o", capsize=3, label=lab, color=col)
    _finish(ax)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"scale ratio $\alpha_A/\alpha_V$")
    ax.set_ylabel("accuracy contribution (masked)")
    ax.set_title("Feature-scale disparity vs modality contribution")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_scale_vs_rho(agg, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x, m, ci = _err(agg, "final_rho")
    ax.errorbar(x, m, yerr=ci, marker="s", capsize=3, color="#2ca02c", label=r"final $\rho$")
    x2, m2, ci2 = _err(agg, "early_rho")
    ax.errorbar(x2, m2, yerr=ci2, marker="^", capsize=3, color="#9467bd",
                label=r"early $\rho$ (epoch 5)", alpha=0.8)
    _finish(ax)
    ax.set_yscale("log")
    ax.set_xlabel(r"scale ratio $\alpha_A/\alpha_V$")
    ax.set_ylabel(r"activation-norm ratio $\rho=E\|h_A\|/E\|h_V\|$")
    ax.set_title("Feature-scale disparity vs activation-norm ratio")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_scale_vs_gain(agg, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x, m, ci = _err(agg, "test_multimodal_gain")
    ax.errorbar(x, m, yerr=ci, marker="D", capsize=3, color="#ff7f0e")
    _finish(ax)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"scale ratio $\alpha_A/\alpha_V$")
    ax.set_ylabel("multimodal gain  (Acc_AV - max(Acc_A, Acc_V))")
    ax.set_title("Feature-scale disparity vs multimodal gain")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_scale_vs_masked_acc(agg, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for key, lab, col in [("test_acc_av", "Acc (A+V)", "#000000"),
                          ("test_acc_a", "Acc (A only)", "#1f77b4"),
                          ("test_acc_v", "Acc (V only)", "#d62728")]:
        x, m, ci = _err(agg, key)
        ax.errorbar(x, m, yerr=ci, marker="o", capsize=3, label=lab, color=col)
    _finish(ax)
    ax.set_xlabel(r"scale ratio $\alpha_A/\alpha_V$")
    ax.set_ylabel("accuracy")
    ax.set_title("Masked-modality accuracy vs scale disparity")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def make_e2_figures(agg, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn, name in [
        (plot_scale_vs_contribution, "e2_scale_vs_contribution.png"),
        (plot_scale_vs_rho, "e2_scale_vs_rho.png"),
        (plot_scale_vs_gain, "e2_scale_vs_gain.png"),
        (plot_scale_vs_masked_acc, "e2_scale_vs_masked_acc.png"),
    ]:
        p = fig_dir / name
        fn(agg, p)
        paths.append(p)
    return paths
