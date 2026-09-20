"""UTD-MHAD one-time caching of ENCODER INPUTS (Phase 4). Valid caching:
depth clips + inertial signals are the encoder INPUTS; encoders run/update every step.

Writes to data/utd_mhad/cache/: {split}_depth.npy [N,T,1,H,W] f16,
{split}_inertial.npy [N,6,L] f16, {split}_y.npy int64, {split}_subject.npy int64,
{split}_keys.npy <U16. Inertial is per-channel z-scored using TRAIN stats only
(written back into configs/utd_mhad.yaml-compatible json). Deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from data.utd_manifest import build_manifest
from data.utd_mhad import load_depth_clip, load_inertial
from msc.runlog import git_hash, load_config


def main():
    cfg = load_config(ROOT / "configs" / "utd_mhad.yaml")
    raw_dir = (ROOT / cfg["data"]["raw_dir"]).resolve()
    split = cfg["data"]["split"]
    nf, hw = cfg["data"]["depth"]["n_frames"], cfg["data"]["depth"]["hw"]
    L = cfg["data"]["inertial"]["length"]
    rows, audit = build_manifest(raw_dir, split)
    print("intersection usable trials:", audit["intersection"], "per_split:", audit["per_split"])

    by = {sp: [r for r in rows if r["split"] == sp] for sp in split}

    # --- load raw inertial for TRAIN to fit per-channel standardization stats ---
    def load_iner_stack(rs):
        return np.stack([load_inertial(r["inertial_path"], L) for r in rs], axis=0)  # [N,6,L]

    tr_iner = load_iner_stack(by["train"])
    ch_mean = tr_iner.mean(axis=(0, 2)).astype(np.float64)   # per-channel
    ch_std = tr_iner.std(axis=(0, 2)).clip(1e-6).astype(np.float64)
    print("inertial TRAIN per-channel mean:", np.round(ch_mean, 4).tolist())
    print("inertial TRAIN per-channel std :", np.round(ch_std, 4).tolist())

    cache = ROOT / cfg["data"]["cache_dir"]; cache.mkdir(parents=True, exist_ok=True)

    def zscore(x):  # x [N,6,L]
        return (x - ch_mean[None, :, None]) / ch_std[None, :, None]

    fails = []
    for sp, rs in by.items():
        dep, ine, y, subj, keys = [], [], [], [], []
        raw_iner = tr_iner if sp == "train" else load_iner_stack(rs)
        ine_z = zscore(raw_iner).astype(np.float16)
        for n, r in enumerate(rs):
            try:
                dep.append(load_depth_clip(r["depth_path"], nf, hw))
            except Exception as e:
                fails.append({"key": r["key"], "err": repr(e)}); continue
            ine.append(ine_z[n]); y.append(r["action"] - 1); subj.append(r["subject"]); keys.append(r["key"])
        dep = np.stack(dep, axis=0).astype(np.float16)       # [N,T,1,H,W]
        ine = np.stack(ine, axis=0).astype(np.float16)       # [N,6,L]
        np.save(cache / f"{sp}_depth.npy", dep)
        np.save(cache / f"{sp}_inertial.npy", ine)
        np.save(cache / f"{sp}_y.npy", np.array(y, dtype=np.int64))
        np.save(cache / f"{sp}_subject.npy", np.array(subj, dtype=np.int64))
        np.save(cache / f"{sp}_keys.npy", np.array(keys))
        print(f"  {sp}: depth{dep.shape} inertial{ine.shape} y{len(y)}")

    stats = {"git": git_hash(), "inertial_channel_mean": ch_mean.tolist(),
             "inertial_channel_std": ch_std.tolist(), "n_frames": nf, "hw": hw, "length": L,
             "n_failed": len(fails), "failed": fails, "per_split": audit["per_split"]}
    json.dump(stats, open(cache / "extract_stats.json", "w"), indent=2)
    print("failed decodes:", len(fails), fails if fails else "")
    print("cache ->", cache)


if __name__ == "__main__":
    main()
