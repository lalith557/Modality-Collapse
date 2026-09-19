"""Fusion operators. Cross-attention is intentionally deferred to an ablation."""
from __future__ import annotations

import torch


def fused_dim(fusion: str, rep_dim_a: int, rep_dim_v: int) -> int:
    if fusion == "sum":
        if rep_dim_a != rep_dim_v:
            raise ValueError("sum fusion requires equal rep dims")
        return rep_dim_a
    if fusion == "concat":
        return rep_dim_a + rep_dim_v
    raise ValueError(f"unknown fusion '{fusion}'")


def fuse(fusion: str, h_a: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
    if fusion == "sum":
        return h_a + h_v
    if fusion == "concat":
        return torch.cat([h_a, h_v], dim=1)
    raise ValueError(f"unknown fusion '{fusion}'")
