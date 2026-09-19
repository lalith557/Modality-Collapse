"""Per-modality normalization modules (the candidate 'minimal cure', E3).

Options:
    none      -> identity
    layernorm -> nn.LayerNorm over the representation dim (affine)
    zscore    -> per-feature standardization using TRAIN statistics only
                 (fit via `fit`; no val/test leakage).

For the E2 milestone the default is 'none'. layernorm/zscore are wired in for E3.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ZScoreNorm(nn.Module):
    """Standardize with fixed train statistics. Identity until `fit` is called."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.fitted = False

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "ZScoreNorm":
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(x.std(dim=0).clamp_min(1e-6))
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def make_norm(kind: str, dim: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "zscore":
        return ZScoreNorm(dim)
    raise ValueError(f"unknown normalization '{kind}'")
