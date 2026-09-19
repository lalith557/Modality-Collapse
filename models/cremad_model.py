"""CREMA-D two-stream model. Mirrors models/multimodal_model.MultimodalModel's
interface (encode / forward(mask_a,mask_v,return_reps)) so the frozen analysis
helpers (measure_rho, contribution_metrics), per-modality normalization, and the
fusion-boundary scale injection are reused UNCHANGED.

Modality A = AUDIO, modality V = VIDEO (so Delta_A = Acc_AV - Acc_video = audio
contribution, matching PREREGISTRATION_CREMAD.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from interventions.per_modality_norm import make_norm
from interventions.scale_injection import ScaleInjector
from models.cremad_encoders import AudioCNN, VideoCNN
from models.fusion import fuse, fused_dim
from models.heads import make_head


class CremadMultimodalModel(nn.Module):
    def __init__(self, n_classes: int, rep_dim: int = 64, fusion: str = "sum",
                 normalization: str = "none", head: str = "linear",
                 scale_injection: dict | None = None):
        super().__init__()
        self.fusion = fusion
        self.enc_a = AudioCNN(rep_dim)     # audio
        self.enc_v = VideoCNN(rep_dim)     # video
        self.norm_a = make_norm(normalization, rep_dim)
        self.norm_v = make_norm(normalization, rep_dim)
        si = scale_injection or {}
        self.scale = ScaleInjector(si.get("alpha_a", 1.0), si.get("alpha_v", 1.0),
                                   si.get("enabled", False))
        self.head = make_head(head, fused_dim(fusion, rep_dim, rep_dim), n_classes, 64)

    def encode(self, x_a, x_v):
        h_a = self.norm_a(self.enc_a(x_a))
        h_v = self.norm_v(self.enc_v(x_v))
        if self.scale.enabled:
            h_a = self.scale.alpha_a * h_a
            h_v = self.scale.alpha_v * h_v
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
