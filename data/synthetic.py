"""Controlled synthetic multimodal generator (H2 causal environment).

Generative model, per modality m in {a, v}:

    x_m = alpha_m * (mu_{y}^{m} + sigma_m * epsilon),   epsilon ~ N(0, I)

Design invariants that make the causal claim valid:

* Class prototypes ``mu^m`` are generated from a FIXED ``prototype_seed`` and are
  identical across every alpha in a scale sweep -> the *information structure*
  never changes.
* The pre-scale base ``(mu + sigma*eps)`` is drawn from ``data_seed`` which does
  NOT depend on alpha. ``alpha`` is applied as a pure post-multiplication.
  Therefore a linear classifier can absorb ``alpha`` into its weights and its
  accuracy is invariant to ``alpha`` -- magnitude changes, information does not.
* Both modalities share the label ``y`` (they describe the same target) and by
  default use matched (margin, sigma, dim), so any modality dominance that
  emerges is attributable to feature magnitude, not to an information asymmetry.

The `dataset_report` function empirically verifies these invariants (class
balance, norms, scale ratio, and per-modality linear-probe accuracy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Prototypes (fixed information structure)
# --------------------------------------------------------------------------- #
def make_prototypes(n_classes: int, dim: int, margin: float, seed: int) -> np.ndarray:
    """Return (n_classes, dim) class prototypes.

    When ``n_classes <= dim`` we use orthonormal directions so every pair of
    prototypes is equidistant (distance = margin * sqrt(2)); this gives a clean,
    modality-matched Bayes accuracy. Otherwise we fall back to random unit
    directions scaled by ``margin``.
    """
    rng = np.random.default_rng(seed)
    if n_classes <= dim:
        g = rng.standard_normal((dim, n_classes))
        q, _ = np.linalg.qr(g)              # (dim, n_classes) orthonormal columns
        protos = q.T * margin               # (n_classes, dim)
    else:
        g = rng.standard_normal((n_classes, dim))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        protos = g * margin
    return protos.astype(np.float32)


@dataclass
class SyntheticDataset:
    tensors: dict = field(default_factory=dict)   # split -> (a, v, y) torch tensors
    meta: dict = field(default_factory=dict)

    def split(self, name: str):
        return self.tensors[name]


def _sample_split(
    n: int, protos: np.ndarray, sigma: float, alpha: float, rng: np.random.Generator
):
    """Draw one modality's split. Returns (x float32 [n, dim], y int64 [n])."""
    n_classes, dim = protos.shape
    y = rng.integers(0, n_classes, size=n)
    eps = rng.standard_normal((n, dim)).astype(np.float32)
    base = protos[y] + sigma * eps          # mu_y + sigma * epsilon
    x = alpha * base                         # feature-magnitude scaling
    return x.astype(np.float32), y.astype(np.int64)


def generate_dataset(params: dict[str, Any]) -> SyntheticDataset:
    """Generate a controlled multimodal dataset.

    Expected keys in ``params`` (see configs/synthetic_causal.yaml -> data):
        n_classes, n_dim_a, n_dim_v, prototype_margin, sigma_a, sigma_v,
        alpha_a, alpha_v, n_train, n_val, n_test, prototype_seed
    Plus:
        data_seed : controls the noise/label draws (independent of alpha).

    The two modalities share labels split-by-split (paired samples).
    """
    p = params
    n_classes = int(p["n_classes"])
    da, dv = int(p["n_dim_a"]), int(p["n_dim_v"])
    margin = float(p["prototype_margin"])
    sa, sv = float(p["sigma_a"]), float(p["sigma_v"])
    aa, av = float(p["alpha_a"]), float(p["alpha_v"])
    proto_seed = int(p["prototype_seed"])
    data_seed = int(p.get("data_seed", 0))

    protos_a = make_prototypes(n_classes, da, margin, proto_seed)
    protos_v = make_prototypes(n_classes, dv, margin, proto_seed + 1)

    sizes = {"train": int(p["n_train"]), "val": int(p["n_val"]), "test": int(p["n_test"])}
    tensors: dict = {}
    # One RNG per split, keyed off data_seed only (NOT alpha) so the base draw is
    # identical across a scale sweep. Labels are shared across modalities within a
    # split by drawing y once, then both modalities' features conditioned on it.
    for si, (name, n) in enumerate(sizes.items()):
        rng = np.random.default_rng(data_seed * 100 + si)
        y = rng.integers(0, n_classes, size=n).astype(np.int64)
        eps_a = rng.standard_normal((n, da)).astype(np.float32)
        eps_v = rng.standard_normal((n, dv)).astype(np.float32)
        xa = (aa * (protos_a[y] + sa * eps_a)).astype(np.float32)
        xv = (av * (protos_v[y] + sv * eps_v)).astype(np.float32)
        tensors[name] = (
            torch.from_numpy(xa),
            torch.from_numpy(xv),
            torch.from_numpy(y),
        )

    meta = {
        "n_classes": n_classes, "n_dim_a": da, "n_dim_v": dv,
        "prototype_margin": margin, "sigma_a": sa, "sigma_v": sv,
        "alpha_a": aa, "alpha_v": av, "alpha_ratio": aa / av,
        "prototype_seed": proto_seed, "data_seed": data_seed,
        "sizes": sizes,
        "snr_a": margin / sa, "snr_v": margin / sv,   # theoretical (scale-invariant)
    }
    return SyntheticDataset(tensors=tensors, meta=meta)


# --------------------------------------------------------------------------- #
# Reporting / verification of the causal invariants
# --------------------------------------------------------------------------- #
def dataset_report(ds: SyntheticDataset, run_probe: bool = True,
                   device: torch.device | None = None) -> dict:
    """Empirically characterise a generated dataset.

    Verifies: class balance, per-modality input norms, the empirical scale
    ratio, and (optionally) per-modality linear-probe accuracy -- the key check
    that information is held constant when only alpha changes.
    """
    rep: dict = {"meta": ds.meta}
    for name, (a, v, y) in ds.tensors.items():
        counts = torch.bincount(y, minlength=ds.meta["n_classes"]).tolist()
        na = a.norm(dim=1)
        nv = v.norm(dim=1)
        rep[name] = {
            "n": int(y.numel()),
            "class_counts": counts,
            "class_balance_min_frac": min(counts) / max(1, sum(counts)),
            "mean_norm_a": float(na.mean()), "mean_norm_v": float(nv.mean()),
            "empirical_scale_ratio": float(na.mean() / nv.mean()),
        }
    if run_probe:
        from analysis.probes import linear_probe_accuracy  # lazy import
        atr, vtr, ytr = ds.tensors["train"]
        ate, vte, yte = ds.tensors["test"]
        n_classes = ds.meta["n_classes"]
        rep["probe"] = {
            "acc_a": linear_probe_accuracy(atr, ytr, ate, yte, n_classes, device),
            "acc_v": linear_probe_accuracy(vtr, ytr, vte, yte, n_classes, device),
        }
    return rep
