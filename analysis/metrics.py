"""Evaluation metrics: modality masking, contributions, multimodal gain, macro-F1.

Masking convention (representation-level, the standard modality-dropout probe):
    both   : both modalities active
    a_only : modality V representation zeroed  (measures A alone)
    v_only : modality A representation zeroed  (measures V alone)

Contributions:
    Delta_A = Acc_AV - Acc_V   (accuracy A adds on top of V)
    Delta_V = Acc_AV - Acc_A   (accuracy V adds on top of A)
    multimodal_gain = Acc_AV - max(Acc_A, Acc_V)
"""
from __future__ import annotations

import torch


@torch.no_grad()
def macro_f1(preds: torch.Tensor, targets: torch.Tensor, n_classes: int) -> float:
    """Macro-averaged F1 computed on-device (GPU)."""
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp / denom) if denom > 0 else torch.tensor(0.0, device=preds.device))
    return float(torch.stack(f1s).mean())


_MASKS = {"both": (1.0, 1.0), "a_only": (1.0, 0.0), "v_only": (0.0, 1.0)}


@torch.no_grad()
def evaluate_with_mask(model, x_a, x_v, y, mode: str, n_classes: int) -> dict:
    was_training = model.training
    model.eval()
    ma, mv = _MASKS[mode]
    logits = model(x_a, x_v, mask_a=ma, mask_v=mv)
    preds = logits.argmax(1)
    acc = float((preds == y).float().mean())
    f1 = macro_f1(preds, y, n_classes)
    if was_training:
        model.train()
    return {"acc": acc, "macro_f1": f1}


@torch.no_grad()
def contribution_metrics(model, x_a, x_v, y, n_classes: int) -> dict:
    both = evaluate_with_mask(model, x_a, x_v, y, "both", n_classes)
    a_only = evaluate_with_mask(model, x_a, x_v, y, "a_only", n_classes)
    v_only = evaluate_with_mask(model, x_a, x_v, y, "v_only", n_classes)
    acc_av, acc_a, acc_v = both["acc"], a_only["acc"], v_only["acc"]
    return {
        "acc_av": acc_av, "acc_a": acc_a, "acc_v": acc_v,
        "f1_av": both["macro_f1"], "f1_a": a_only["macro_f1"], "f1_v": v_only["macro_f1"],
        "delta_a": acc_av - acc_v,           # contribution of A
        "delta_v": acc_av - acc_a,           # contribution of V
        "multimodal_gain": acc_av - max(acc_a, acc_v),
        # signed dominance: >0 means A dominates (A alone strong, V adds little)
        "dominance_a_minus_v": (acc_a - acc_v),
    }
