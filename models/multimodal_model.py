"""Two-stream multimodal classifier.

Forward pipeline (single documented data path):

    x_a -> enc_a -> norm_a -> scale_a --.
                                        |-- (diagnostic reads reps HERE) -- mask -- fuse -- head -> logits
    x_v -> enc_v -> norm_v -> scale_v --'

* Diagnostic rho is measured on the UNMASKED representations (h_a, h_v).
* Masking multiplies a modality's representation by 0/1 for contribution analysis;
  it does not affect the reps returned for the diagnostic.
* Scale injection is a no-op unless explicitly enabled (E2 drives scale via data).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from interventions.per_modality_norm import make_norm
from interventions.scale_injection import ScaleInjector
from models.encoders import TinyMLPEncoder
from models.fusion import fuse, fused_dim
from models.heads import make_head


class MultimodalModel(nn.Module):
    def __init__(
        self,
        in_dim_a: int, in_dim_v: int, n_classes: int,
        hidden_dim: int = 64, rep_dim: int = 32,
        fusion: str = "sum", normalization: str = "none",
        activation: str = "relu", head: str = "linear",
        scale_injection: dict | None = None,
    ):
        super().__init__()
        self.fusion = fusion
        self.enc_a = TinyMLPEncoder(in_dim_a, hidden_dim, rep_dim, activation)
        self.enc_v = TinyMLPEncoder(in_dim_v, hidden_dim, rep_dim, activation)
        self.norm_a = make_norm(normalization, rep_dim)
        self.norm_v = make_norm(normalization, rep_dim)
        si = scale_injection or {}
        self.scale = ScaleInjector(
            alpha_a=si.get("alpha_a", 1.0),
            alpha_v=si.get("alpha_v", 1.0),
            enabled=si.get("enabled", False),
        )
        self.head = make_head(head, fused_dim(fusion, rep_dim, rep_dim), n_classes, hidden_dim)

    def encode(self, x_a: torch.Tensor, x_v: torch.Tensor):
        """Return the fusion-boundary representations (post norm + scale, pre mask)."""
        h_a = self.scale.alpha_a * self.norm_a(self.enc_a(x_a)) if self.scale.enabled \
            else self.norm_a(self.enc_a(x_a))
        h_v = self.scale.alpha_v * self.norm_v(self.enc_v(x_v)) if self.scale.enabled \
            else self.norm_v(self.enc_v(x_v))
        return h_a, h_v

    def forward(
        self, x_a: torch.Tensor, x_v: torch.Tensor,
        mask_a: float = 1.0, mask_v: float = 1.0, return_reps: bool = False,
    ):
        h_a, h_v = self.encode(x_a, x_v)          # unmasked reps (for diagnostic)
        z = fuse(self.fusion, mask_a * h_a, mask_v * h_v)
        logits = self.head(z)
        if return_reps:
            return logits, (h_a, h_v)
        return logits


def build_model(cfg: dict, in_dim_a: int, in_dim_v: int, n_classes: int) -> MultimodalModel:
    m = cfg["model"]
    return MultimodalModel(
        in_dim_a=in_dim_a, in_dim_v=in_dim_v, n_classes=n_classes,
        hidden_dim=int(m.get("hidden_dim", 64)),
        rep_dim=int(m.get("rep_dim", 32)),
        fusion=m.get("fusion", "sum"),
        normalization=m.get("normalization", "none"),
        activation=m.get("activation", "relu"),
        head=m.get("head", "linear"),
        scale_injection=cfg.get("scale_injection"),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
