"""OGM-GE gradient modulation (Peng et al., CVPR 2022), adapted to this model.

Fidelity (see PREREGISTRATION_E5.md):
- FAITHFUL: per-modality softmax-score discrepancy ratio; dominant-modality
  coefficient k = 1 - tanh(alpha * ratio); modulation applied to the uni-modal
  ENCODER gradients only (head unmodulated).
- PROJECT-SPECIFIC: per-modality logit score for SUM fusion uses the shared head
  weight (s_m = h_m W^T + b), since sum fusion has no modality-split classifier.
- ADAPTED: GE is a Gaussian-noise approximation restoring part of the covariance
  reduced by modulation (the paper maintains covariance more precisely).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def per_modality_scores(model, h_a: torch.Tensor, h_b: torch.Tensor):
    """Per-modality class logits using the (linear) head. Sum or concat fusion."""
    head = model.head
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("OGM-GE here requires a linear head")
    W, b = head.weight, head.bias
    if model.fusion == "sum":
        s_a = h_a @ W.t() + b
        s_b = h_b @ W.t() + b
    elif model.fusion == "concat":
        d = h_a.shape[1]
        s_a = h_a @ W[:, :d].t() + b
        s_b = h_b @ W[:, d:].t() + b
    else:
        raise ValueError(f"unsupported fusion '{model.fusion}'")
    return s_a, s_b


@torch.no_grad()
def ogm_coefficients(s_a, s_b, y, alpha: float):
    """Return (k_a, k_b, ratio_a_over_b). Dominant modality attenuated by
    k = 1 - tanh(alpha * ratio); the weaker modality is left at 1.0."""
    pa = F.softmax(s_a, dim=1); pb = F.softmax(s_b, dim=1)
    idx = torch.arange(y.shape[0], device=y.device)
    score_a = float(pa[idx, y].sum())
    score_b = float(pb[idx, y].sum())
    eps = 1e-8
    ratio = score_a / (score_b + eps)
    if ratio > 1.0:                       # A dominates -> attenuate A
        k_a = 1.0 - math.tanh(alpha * ratio); k_b = 1.0
    elif ratio < 1.0:                     # B dominates -> attenuate B
        k_b = 1.0 - math.tanh(alpha * (1.0 / (ratio + eps))); k_a = 1.0
    else:
        k_a = k_b = 1.0
    return k_a, k_b, ratio


@torch.no_grad()
def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


@torch.no_grad()
def apply_modulation(enc_a_params, enc_b_params, k_a: float, k_b: float,
                     ge_gamma: float = 0.0):
    """Multiply encoder grads by their coefficients; optionally add GE noise.
    Returns (pre_norm_a, pre_norm_b) measured BEFORE modulation."""
    pre_a = _grad_norm(enc_a_params); pre_b = _grad_norm(enc_b_params)
    for params, k in ((enc_a_params, k_a), (enc_b_params, k_b)):
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(k)
            if ge_gamma > 0.0 and k < 1.0:
                std = p.grad.detach().std()
                if torch.isfinite(std) and std > 0:
                    noise = torch.randn_like(p.grad) * (ge_gamma * (1.0 - k) * std)
                    p.grad.add_(noise)
    return pre_a, pre_b
