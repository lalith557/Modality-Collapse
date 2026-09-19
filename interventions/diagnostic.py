"""Activation-norm diagnostic (H1).

Measures rho = E[||h_a||_2] / E[||h_v||_2] at the fusion boundary, where h_a,
h_v are the *unmasked* modality representations (post-encoder, post-norm,
post-scale-injection, pre-fusion). The diagnostic is read-only: it never alters
model behaviour.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def norm_stats(h_a: torch.Tensor, h_v: torch.Tensor) -> dict:
    """Per-batch norm statistics and rho."""
    na = h_a.norm(dim=1)
    nv = h_v.norm(dim=1)
    mean_a = float(na.mean())
    mean_v = float(nv.mean())
    rho = mean_a / mean_v if mean_v > 0 else float("inf")
    return {
        "mean_norm_a": mean_a,
        "mean_norm_v": mean_v,
        "var_norm_a": float(na.var(unbiased=False)),
        "var_norm_v": float(nv.var(unbiased=False)),
        "p50_norm_a": float(na.median()),
        "p50_norm_v": float(nv.median()),
        "rho": rho,
        "log_rho": math.log(rho) if rho not in (0.0, float("inf")) else float("nan"),
    }


@torch.no_grad()
def measure_rho(model, x_a: torch.Tensor, x_v: torch.Tensor) -> dict:
    """Compute norm stats over a (whole) evaluation batch using the model's reps."""
    was_training = model.training
    model.eval()
    _, reps = model(x_a, x_v, return_reps=True)
    h_a, h_v = reps
    stats = norm_stats(h_a, h_v)
    if was_training:
        model.train()
    return stats


class NormTracker:
    """Accumulates per-epoch norm/rho stats into a list of dicts."""

    def __init__(self):
        self.history: list[dict] = []

    def record(self, epoch: int, stats: dict) -> None:
        row = {"epoch": epoch}
        row.update(stats)
        self.history.append(row)

    def at_epoch(self, epoch: int) -> dict | None:
        for row in self.history:
            if row["epoch"] == epoch:
                return row
        return None
