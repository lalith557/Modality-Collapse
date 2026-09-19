"""Tests for the model, masking, rho diagnostic, and a tiny training smoke run."""
import torch

from analysis.metrics import contribution_metrics, evaluate_with_mask, macro_f1
from data.synthetic import generate_dataset
from interventions.diagnostic import measure_rho
from models.multimodal_model import MultimodalModel
from msc.runlog import load_config
from pathlib import Path


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA = dict(
    n_classes=5, n_dim_a=16, n_dim_v=16, prototype_margin=2.5,
    sigma_a=1.0, sigma_v=1.0, alpha_a=1.0, alpha_v=1.0,
    n_train=800, n_val=300, n_test=600, prototype_seed=1, data_seed=0,
)


def test_gpu_available():
    # Project requirement: experiments run on CUDA.
    assert torch.cuda.is_available(), "CUDA not available"


def test_forward_and_reps_shapes():
    dev = _dev()
    m = MultimodalModel(16, 16, 5, rep_dim=16, fusion="sum").to(dev)
    x = torch.randn(32, 16, device=dev)
    logits, (ha, hv) = m(x, x, return_reps=True)
    assert logits.shape == (32, 5)
    assert ha.shape == (32, 16) and hv.shape == (32, 16)


def test_masking_zeroes_modality():
    dev = _dev()
    torch.manual_seed(0)
    m = MultimodalModel(16, 16, 5, rep_dim=16, fusion="sum").to(dev)
    xa = torch.randn(8, 16, device=dev)
    xv = torch.randn(8, 16, device=dev)
    # a_only must be independent of xv (V masked out for sum fusion)
    la = m(xa, xv, mask_a=1.0, mask_v=0.0)
    lb = m(xa, torch.randn_like(xv), mask_a=1.0, mask_v=0.0)
    assert torch.allclose(la, lb, atol=1e-5)


def test_macro_f1_bounds():
    preds = torch.tensor([0, 1, 2, 2, 1])
    tgts = torch.tensor([0, 1, 2, 0, 1])
    f1 = macro_f1(preds, tgts, 3)
    assert 0.0 <= f1 <= 1.0


def test_rho_tracks_injected_scale():
    """With rep-level scale injection enabled, rho ~ alpha_a/alpha_v."""
    dev = _dev()
    m = MultimodalModel(16, 16, 5, rep_dim=16, fusion="sum",
                        scale_injection={"enabled": True, "alpha_a": 4.0, "alpha_v": 1.0}).to(dev)
    x = torch.randn(256, 16, device=dev)
    stats = measure_rho(m, x, x)
    assert 3.0 < stats["rho"] < 5.0


def test_tiny_training_reduces_loss():
    dev = _dev()
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "synthetic_causal.yaml")
    cfg["train"]["epochs"] = 5
    cfg["data"].update(DATA)
    from experiments.train import train_model
    ds = generate_dataset({**cfg["data"], "alpha_a": 1.0, "alpha_v": 1.0, "data_seed": 0})
    res = train_model(cfg, ds, dev, seed=0)
    # both modalities should contribute in the balanced case
    assert res["test"]["acc_av"] > 0.4
    assert res["test"]["delta_a"] > 0.0 and res["test"]["delta_v"] > 0.0
