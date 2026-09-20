"""UTD-MHAD two-stream model (depth + inertial).

Mirrors models/cremad_model.CremadMultimodalModel's interface EXACTLY (raw_encode
/ encode / forward(mask_a,mask_v,return_reps), scale_first ordering) so the frozen
analysis helpers (measure_rho, contribution_metrics, rho_encoder), per-modality
normalization, and the fusion-boundary ScaleInjector are reused UNCHANGED.

Modality A = DEPTH (enc_a), Modality B = INERTIAL (enc_v). Therefore, in the reused
metric code: dominance = Acc_depth - Acc_inertial, Delta_depth = Acc_AB - Acc_inertial,
Delta_inertial = Acc_AB - Acc_depth (i.e. 'audio'->depth, 'video'->inertial).
"""
from __future__ import annotations

import torch.nn as nn

from interventions.per_modality_norm import make_norm
from interventions.scale_injection import ScaleInjector
from models.fusion import fuse, fused_dim
from models.heads import make_head
from models.utd_encoders import DepthCNN, InertialCNN


class UTDMultimodalModel(nn.Module):
    def __init__(self, n_classes: int, rep_dim: int = 64, fusion: str = "sum",
                 normalization: str = "none", head: str = "linear",
                 scale_injection: dict | None = None, scale_first: bool = False,
                 in_ch: int = 6):
        super().__init__()
        self.fusion = fusion
        # scale_first=True -> enc -> scale -> norm (so z-score can remove imposed scale; used here).
        self.scale_first = scale_first
        self.enc_a = DepthCNN(rep_dim)          # A = depth
        self.enc_v = InertialCNN(rep_dim, in_ch)  # B = inertial
        self.norm_a = make_norm(normalization, rep_dim)
        self.norm_v = make_norm(normalization, rep_dim)
        si = scale_injection or {}
        self.scale = ScaleInjector(si.get("alpha_a", 1.0), si.get("alpha_v", 1.0),
                                   si.get("enabled", False))
        self.head = make_head(head, fused_dim(fusion, rep_dim, rep_dim), n_classes, 64)

    def raw_encode(self, x_a, x_v):
        """Encoder outputs BEFORE norm/scale (the 'learned' representation)."""
        return self.enc_a(x_a), self.enc_v(x_v)

    def encode(self, x_a, x_v):
        h_a = self.enc_a(x_a); h_v = self.enc_v(x_v)
        if self.scale_first:
            if self.scale.enabled:
                h_a = self.scale.alpha_a * h_a; h_v = self.scale.alpha_v * h_v
            h_a = self.norm_a(h_a); h_v = self.norm_v(h_v)
        else:
            h_a = self.norm_a(h_a); h_v = self.norm_v(h_v)
            if self.scale.enabled:
                h_a = self.scale.alpha_a * h_a; h_v = self.scale.alpha_v * h_v
        return h_a, h_v

    def forward(self, x_a, x_v, mask_a: float = 1.0, mask_v: float = 1.0, return_reps: bool = False):
        h_a, h_v = self.encode(x_a, x_v)
        z = fuse(self.fusion, mask_a * h_a, mask_v * h_v)
        logits = self.head(z)
        if return_reps:
            return logits, (h_a, h_v)
        return logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
