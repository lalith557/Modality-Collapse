"""Reusable GPU training loop for the multimodal model.

All tensors live on the GPU. Records the per-epoch activation-norm diagnostic
(rho) on the validation split, snapshots the pre-registered early-epoch rho, and
reports final contribution metrics on the test split.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from analysis.metrics import contribution_metrics, evaluate_with_mask
from interventions.diagnostic import NormTracker, measure_rho
from models.multimodal_model import build_model, count_params
from msc.seed import set_seed


def _optimizer(model, cfg, device):
    t = cfg["train"]
    lr = float(t["lr"])
    lr_a = t.get("lr_a") or lr
    lr_v = t.get("lr_v") or lr
    wd = float(t.get("weight_decay", 0.0))
    enc_a_ids = {id(p) for p in model.enc_a.parameters()}
    enc_v_ids = {id(p) for p in model.enc_v.parameters()}
    rest = [p for p in model.parameters()
            if id(p) not in enc_a_ids and id(p) not in enc_v_ids]
    groups = [
        {"params": list(model.enc_a.parameters()), "lr": float(lr_a)},
        {"params": list(model.enc_v.parameters()), "lr": float(lr_v)},
        {"params": rest, "lr": lr},
    ]
    name = t.get("optimizer", "adam").lower()
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"unknown optimizer '{name}'")


def _to_device(ds, device):
    out = {}
    for name, (a, v, y) in ds.tensors.items():
        out[name] = (a.to(device), v.to(device), y.to(device))
    return out


def train_model(cfg: dict, ds, device: torch.device, seed: int) -> dict:
    """Train once. Returns a result dict with metrics, rho history, provenance."""
    set_seed(seed, deterministic=cfg.get("seed_policy", {}).get("deterministic", True))

    n_classes = ds.meta["n_classes"]
    data = _to_device(ds, device)
    a_tr, v_tr, y_tr = data["train"]
    a_va, v_va, y_va = data["val"]
    a_te, v_te, y_te = data["test"]

    model = build_model(cfg, ds.meta["n_dim_a"], ds.meta["n_dim_v"], n_classes).to(device)

    # Fit zscore normalization on TRAIN reps only (no leakage), if requested.
    if cfg["model"].get("normalization") == "zscore":
        with torch.no_grad():
            h_a, h_v = model.encode(a_tr, v_tr)
            if hasattr(model.norm_a, "fit"):
                model.norm_a.fit(h_a)
                model.norm_v.fit(h_v)

    opt = _optimizer(model, cfg, device)
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(cfg["train"]["epochs"])
    bs = int(cfg["train"]["batch_size"])
    early_epoch = int(cfg.get("diagnostic", {}).get("early_epoch", 5))
    n = y_tr.numel()

    tracker = NormTracker()
    early_rho = None
    t0 = time.time()

    def _epoch_row(ep):
        """rho stats + (cheap) val contribution metrics for the trajectory log."""
        row = dict(measure_rho(model, a_va, v_va))
        c = contribution_metrics(model, a_va, v_va, y_va, n_classes)
        row.update({"val_acc_av": c["acc_av"], "val_delta_a": c["delta_a"],
                    "val_delta_v": c["delta_v"], "val_gain": c["multimodal_gain"],
                    "val_f1_av": c["f1_av"]})
        return row

    # epoch 0 diagnostic (at initialization, before any update)
    tracker.record(0, _epoch_row(0))

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = model(a_tr[idx], v_tr[idx])
            loss = loss_fn(logits, y_tr[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            opt.step()

        stats = _epoch_row(epoch)
        tracker.record(epoch, stats)
        if epoch == early_epoch:
            early_rho = stats

    runtime = time.time() - t0
    final = contribution_metrics(model, a_te, v_te, y_te, n_classes)
    final_val = contribution_metrics(model, a_va, v_va, y_va, n_classes)

    return {
        "seed": seed,
        "alpha_ratio": ds.meta["alpha_ratio"],
        "alpha_a": ds.meta["alpha_a"], "alpha_v": ds.meta["alpha_v"],
        "n_params": count_params(model),
        "runtime_sec": runtime,
        "final_rho": tracker.at_epoch(epochs)["rho"],
        "early_epoch": early_epoch,
        "early_rho": (early_rho or {}).get("rho"),
        "early_log_rho": (early_rho or {}).get("log_rho"),
        "test": final,
        "val": final_val,
        "rho_history": tracker.history,
    }
