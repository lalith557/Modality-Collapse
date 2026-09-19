"""Representation-level scale injection at the fusion boundary.

    h_a <- alpha_a * h_a
    h_v <- alpha_v * h_v

This is applied at a single, documented location: immediately after the
encoders (and normalization), before masking and fusion. In E2 the scale
disparity is driven by the DATA (data/synthetic.py alpha), so this module is
disabled by default (alpha=1). It exists as an independent knob for controlled
representation-level experiments and lives here so the injection location is
unambiguous and logged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScaleInjector(nn.Module):
    def __init__(self, alpha_a: float = 1.0, alpha_v: float = 1.0, enabled: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        # buffers so the actual applied values move with the module / get saved
        self.register_buffer("alpha_a", torch.tensor(float(alpha_a)))
        self.register_buffer("alpha_v", torch.tensor(float(alpha_v)))

    def forward(self, h_a: torch.Tensor, h_v: torch.Tensor):
        if not self.enabled:
            return h_a, h_v
        return self.alpha_a * h_a, self.alpha_v * h_v

    def extra_repr(self) -> str:
        return f"enabled={self.enabled}, alpha_a={float(self.alpha_a)}, alpha_v={float(self.alpha_v)}"
