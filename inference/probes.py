"""Exportable frozen-encoder linear probe.

`analysis/probes.py::linear_probe_accuracy` is the paper's probe, but it returns only an
accuracy. This module performs the *identical* fitting procedure (train-statistic
standardization, nn.Linear, Adam lr=0.05, weight_decay=1e-4, 400 full-batch steps) and
returns the fitted parameters so a unimodal prediction can be made from a classifier
trained on that modality alone. Parity with the original function is tested in
tests/studio/test_probes.py.
"""
from __future__ import annotations

import numpy as np
import torch

PROBE_STEPS = 400
PROBE_LR = 0.05
PROBE_WEIGHT_DECAY = 1e-4


def fit_linear_probe(x_train: torch.Tensor, y_train: torch.Tensor, n_classes: int,
                     device: torch.device, steps: int = PROBE_STEPS, lr: float = PROBE_LR,
                     weight_decay: float = PROBE_WEIGHT_DECAY) -> dict:
    """Mirror of analysis.probes.linear_probe_accuracy, returning the classifier."""
    xtr = x_train.to(device).float()
    ytr = y_train.to(device).long()
    mu = xtr.mean(dim=0, keepdim=True)
    sd = xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    xtr = (xtr - mu) / sd
    clf = torch.nn.Linear(xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    clf.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(clf(xtr), ytr)
        loss.backward()
        opt.step()
    clf.eval()
    return {"weight": clf.weight.detach().cpu().numpy().astype(np.float32),
            "bias": clf.bias.detach().cpu().numpy().astype(np.float32),
            "mu": mu.detach().cpu().numpy().astype(np.float32),
            "sd": sd.detach().cpu().numpy().astype(np.float32)}


def probe_logits(params: dict, x: torch.Tensor) -> torch.Tensor:
    """Apply a fitted probe to representations x [B, D] (any device)."""
    dev = x.device
    mu = torch.as_tensor(params["mu"], device=dev)
    sd = torch.as_tensor(params["sd"], device=dev)
    w = torch.as_tensor(params["weight"], device=dev)
    b = torch.as_tensor(params["bias"], device=dev)
    return ((x.float() - mu) / sd) @ w.T + b
