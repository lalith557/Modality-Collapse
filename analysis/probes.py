"""GPU linear probe for measuring per-modality linear information.

A linear classifier's achievable accuracy is invariant under a global feature
scaling (w . (alpha x) = (alpha w) . x). We standardise features using TRAIN
statistics only (an invertible linear transform, so it does not change the
achievable accuracy) purely to make the optimisation well-conditioned and
alpha-invariant. The reported test accuracy is therefore a stable estimate of
the *linear information* in a modality -- the quantity that must stay constant
across a scale sweep for the H2 causal claim to be valid.

Runs on CUDA (GPU-first, per project requirement).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _standardize_fit(x: torch.Tensor):
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, sd


def linear_probe_accuracy(
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_test: torch.Tensor, y_test: torch.Tensor,
    n_classes: int, device: torch.device | None = None,
    steps: int = 400, lr: float = 0.05, weight_decay: float = 1e-4,
) -> float:
    """Best-achievable multinomial-logistic test accuracy (GPU)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr = x_train.to(device).float()
    xte = x_test.to(device).float()
    ytr = y_train.to(device).long()
    yte = y_test.to(device).long()

    mu, sd = _standardize_fit(xtr)
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd

    dim = xtr.shape[1]
    clf = torch.nn.Linear(dim, n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    clf.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(clf(xtr), ytr)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        acc = (clf(xte).argmax(1) == yte).float().mean().item()
    return acc
