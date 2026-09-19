"""Tiny per-modality encoders: Linear -> ReLU -> Linear."""
from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class TinyMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, rep_dim: int, activation: str = "relu"):
        super().__init__()
        act = _ACT[activation]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, rep_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
