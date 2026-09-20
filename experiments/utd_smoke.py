"""UTD-MHAD model/scale/normalization mechanical smoke test (Phase 5).

Checks (all must PASS before the full run):
  1. both modalities load from cache with expected dims;
  2. param counts for both encoders (matched-capacity);
  3. forward + backward pass produce finite loss and gradients on GPU;
  4. scale x10 mechanically changes rho_fusion in the expected direction
     (depth x10 -> rho ~10x baseline; inertial x10 -> rho ~0.1x baseline);
  5. per-modality z-score removes the imposed scale (rho_fusion -> ~1).
No full training.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from experiments.cremad_run import _chunks, batched_reps, contribution_metrics, measure_rho
from models.utd_model import UTDMultimodalModel, count_params
from msc.device import device_report, get_device
from msc.runlog import load_config
from msc.seed import set_seed


def load_split(cache, sp, device):
    d = torch.from_numpy(np.load(cache / f"{sp}_depth.npy")).float().to(device)
    i = torch.from_numpy(np.load(cache / f"{sp}_inertial.npy")).float().to(device)
    y = torch.from_numpy(np.load(cache / f"{sp}_y.npy")).long().to(device)
    return d, i, y


def main():
    cfg = load_config(ROOT / "configs" / "utd_mhad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    cache = ROOT / cfg["data"]["cache_dir"]
    nc = cfg["data"]["n_classes"]; rep = cfg["model"]["rep_dim"]
    set_seed(0, True)
    d_tr, i_tr, y_tr = load_split(cache, "train", device)
    print(f"[1] shapes: depth {tuple(d_tr.shape)} inertial {tuple(i_tr.shape)} y {tuple(y_tr.shape)} "
          f"classes={int(y_tr.min())}..{int(y_tr.max())}")
    results = {}

    # [2] param counts
    m = UTDMultimodalModel(nc, rep_dim=rep, fusion=cfg["model"]["fusion"], head=cfg["model"]["head"]).to(device)
    pa, pv, pt = count_params(m.enc_a), count_params(m.enc_v), count_params(m)
    ratio = pa / pv
    print(f"[2] params: depth-enc={pa} inertial-enc={pv} ratio={ratio:.3f} total={pt}")
    results["params_matched"] = 0.6 < ratio < 1.7

    # [3] forward + backward
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    bs = 16
    logits = m(d_tr[:bs], i_tr[:bs])
    loss = nn.CrossEntropyLoss()(logits, y_tr[:bs])
    loss.backward()
    ga = sum(float(p.grad.pow(2).sum()) for p in m.enc_a.parameters()) ** 0.5
    gv = sum(float(p.grad.pow(2).sum()) for p in m.enc_v.parameters()) ** 0.5
    print(f"[3] forward logits {tuple(logits.shape)} loss={float(loss):.4f} finite={torch.isfinite(loss).item()} "
          f"grad_depth={ga:.3e} grad_inertial={gv:.3e}")
    results["forward_backward"] = (logits.shape == (bs, nc)) and torch.isfinite(loss).item() and ga > 0 and gv > 0

    # [4] scale mechanics (same weights; toggle scale). scale_first, norm=none.
    m2 = UTDMultimodalModel(nc, rep_dim=rep, fusion="sum", normalization="none",
                            scale_injection={"enabled": False, "alpha_a": 1.0, "alpha_v": 1.0},
                            scale_first=True).to(device)
    m2.eval()
    rho0 = measure_rho(m2, d_tr, i_tr)["rho"]
    m2.scale.enabled = True
    m2.scale.alpha_a.fill_(10.0); m2.scale.alpha_v.fill_(1.0)
    rho_d10 = measure_rho(m2, d_tr, i_tr)["rho"]
    m2.scale.alpha_a.fill_(1.0); m2.scale.alpha_v.fill_(10.0)
    rho_i10 = measure_rho(m2, d_tr, i_tr)["rho"]
    r_up, r_dn = rho_d10 / rho0, rho_i10 / rho0
    print(f"[4] rho baseline={rho0:.4f}  depthx10 rho={rho_d10:.4f} (x{r_up:.2f})  "
          f"inertialx10 rho={rho_i10:.4f} (x{r_dn:.3f})")
    results["scale_mechanics"] = (abs(r_up - 10.0) < 0.5) and (abs(r_dn - 0.1) < 0.05)

    # [5] z-score removes imposed scale
    mz = UTDMultimodalModel(nc, rep_dim=rep, fusion="sum", normalization="zscore",
                            scale_injection={"enabled": True, "alpha_a": 10.0, "alpha_v": 1.0},
                            scale_first=True).to(device)
    with torch.no_grad():
        h_a, h_v = batched_reps(mz, d_tr, i_tr)   # scaled enc outputs (pre-fit norm=identity)
        mz.norm_a.fit(h_a); mz.norm_v.fit(h_v)
    rho_z = measure_rho(mz, d_tr, i_tr)["rho"]
    print(f"[5] z-score rho_fusion under depthx10 = {rho_z:.4f} (expect ~1)")
    results["zscore_removes_scale"] = 0.4 < rho_z < 2.5

    allpass = all(results.values())
    print("\nSMOKE RESULTS:", results)
    print("ALL PASS" if allpass else "FAIL: " + str([k for k, v in results.items() if not v]))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
