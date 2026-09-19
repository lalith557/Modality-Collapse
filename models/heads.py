"""Classifier heads on the fused representation."""
from __future__ import annotations

import torch
import torch.nn as nn


def make_head(kind: str, in_dim: int, n_classes: int, hidden_dim: int = 64) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_dim, n_classes)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes)
        )
    raise ValueError(f"unknown head '{kind}'")
