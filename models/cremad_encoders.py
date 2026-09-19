"""Lightweight, capacity-matched CREMA-D encoders (RTX 4070 / 8 GB)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))


class AudioCNN(nn.Module):
    """log-mel [B,1,64,T] -> rep_dim."""
    def __init__(self, rep_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 32),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(32, rep_dim)

    def forward(self, x):
        return self.fc(self.net(x))


class VideoCNN(nn.Module):
    """frames [B,N,1,64,64] -> per-frame CNN -> temporal mean-pool -> rep_dim.
    Matched conv budget to AudioCNN (Phase-15 capacity control)."""
    def __init__(self, rep_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 32),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(32, rep_dim)

    def forward(self, x):
        b, n = x.shape[:2]
        x = x.reshape(b * n, 1, x.shape[-2], x.shape[-1])
        h = self.net(x).reshape(b, n, -1).mean(dim=1)      # temporal mean-pool
        return self.fc(h)
