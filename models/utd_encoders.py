"""Lightweight, capacity-matched UTD-MHAD encoders (RTX 4070 / 8 GB).

Modality A = DEPTH (Kinect optical depth clip), Modality B = INERTIAL (wearable
IMU 6-channel signal). Depth encoder mirrors the frozen H6 VideoCNN (per-frame
2D CNN + temporal mean-pool); inertial encoder is a matched-budget 1D CNN.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block2d(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))


def _block1d(cin, cout, k=5):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2))


class DepthCNN(nn.Module):
    """depth clip [B,T,1,H,W] -> per-frame 2D CNN -> temporal mean-pool -> rep_dim.
    Identical conv budget to the frozen H6 VideoCNN (16160 params @ rep_dim=64)."""
    def __init__(self, rep_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(_block2d(1, 16), _block2d(16, 32), _block2d(32, 32),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(32, rep_dim)

    def forward(self, x):
        b, n = x.shape[:2]
        x = x.reshape(b * n, 1, x.shape[-2], x.shape[-1])
        h = self.net(x).reshape(b, n, -1).mean(dim=1)      # temporal mean-pool
        return self.fc(h)


class InertialCNN(nn.Module):
    """inertial signal [B,6,L] -> 1D CNN -> global avg-pool -> rep_dim.
    Channels chosen to keep capacity closely matched to DepthCNN (~within 15%)."""
    def __init__(self, rep_dim: int = 64, in_ch: int = 6):
        super().__init__()
        self.net = nn.Sequential(_block1d(in_ch, 16), _block1d(16, 32), _block1d(32, 64),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fc = nn.Linear(64, rep_dim)

    def forward(self, x):
        return self.fc(self.net(x))
