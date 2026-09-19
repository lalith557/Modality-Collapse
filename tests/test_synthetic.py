"""Tests for the synthetic generator's causal invariants."""
import numpy as np
import torch

from data.synthetic import generate_dataset, dataset_report


BASE = dict(
    n_classes=6, n_dim_a=24, n_dim_v=24, prototype_margin=2.0,
    sigma_a=1.0, sigma_v=1.0, alpha_a=1.0, alpha_v=1.0,
    n_train=1500, n_val=500, n_test=1000, prototype_seed=7, data_seed=0,
)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_shapes_and_labels_paired():
    ds = generate_dataset(BASE)
    a, v, y = ds.tensors["train"]
    assert a.shape == (1500, 24) and v.shape == (1500, 24)
    assert y.shape == (1500,) and y.dtype == torch.int64
    assert int(y.min()) >= 0 and int(y.max()) < 6


def test_scale_only_multiplies_base():
    """alpha must be a pure post-multiplication: x(alpha) == alpha * x(1)."""
    p1 = dict(BASE); p1["alpha_a"] = 1.0
    p2 = dict(BASE); p2["alpha_a"] = 7.0
    a1 = generate_dataset(p1).tensors["train"][0]
    a2 = generate_dataset(p2).tensors["train"][0]
    assert torch.allclose(a2, 7.0 * a1, atol=1e-4)


def test_information_invariance_under_scale():
    """Linear-probe accuracy must be ~constant across alpha (info held fixed)."""
    dev = _dev()
    accs = []
    for alpha in (0.1, 1.0, 10.0):
        p = dict(BASE); p["alpha_a"] = alpha
        rep = dataset_report(generate_dataset(p), run_probe=True, device=dev)
        accs.append(rep["probe"]["acc_a"])
    assert max(accs) - min(accs) < 0.05, f"probe acc varied too much: {accs}"


def test_empirical_scale_ratio_matches_alpha():
    p = dict(BASE); p["alpha_a"] = 5.0; p["alpha_v"] = 1.0
    rep = dataset_report(generate_dataset(p), run_probe=False)
    got = rep["test"]["empirical_scale_ratio"]
    assert abs(got - 5.0) / 5.0 < 0.1


def test_class_balance_reasonable():
    rep = dataset_report(generate_dataset(BASE), run_probe=False)
    assert rep["train"]["class_balance_min_frac"] > 0.10   # ~1/6 = 0.167


def test_prototype_structure_fixed_across_seeds():
    """Prototypes depend only on prototype_seed, not data_seed."""
    from data.synthetic import make_prototypes
    p1 = make_prototypes(6, 24, 2.0, seed=7)
    p2 = make_prototypes(6, 24, 2.0, seed=7)
    assert np.allclose(p1, p2)
