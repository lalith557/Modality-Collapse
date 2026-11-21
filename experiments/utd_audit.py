"""UTD-MHAD data integrity audit (Phase 3). Writes DATA_AUDIT_UTD_MHAD.md.

Verifies: depth/inertial trial counts, intersection (usable trials), depth-only /
inertial-only discrepancies, subject-disjoint split, per-subject/per-action counts,
and decodes one depth + one inertial sample to confirm shapes. No training.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.utd_manifest import build_manifest
from data.utd_mhad import load_depth_clip, load_inertial
from msc.runlog import git_hash, load_config

EXPECTED_USABLE = 861


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: allow ∩ etc.
    except Exception:
        pass
    cfg = load_config(ROOT / "configs" / "utd_mhad.yaml")
    raw_dir = (ROOT / cfg["data"]["raw_dir"]).resolve()
    split = cfg["data"]["split"]
    rows, audit = build_manifest(raw_dir, split)
    out = ROOT / cfg["output"]["root"]; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "utd_manifest.csv", index=False)

    # split-disjointness (by subject)
    subs = {sp: set(df[df.split == sp].subject.unique()) for sp in split}
    overlaps = {
        "train_val": sorted(subs["train"] & subs["val"]),
        "train_test": sorted(subs["train"] & subs["test"]),
        "val_test": sorted(subs["val"] & subs["test"]),
    }
    disjoint = all(len(v) == 0 for v in overlaps.values())

    # decode one sample of each modality to confirm shapes
    shapes = {}
    try:
        d0 = load_depth_clip(rows[0]["depth_path"], cfg["data"]["depth"]["n_frames"], cfg["data"]["depth"]["hw"])
        i0 = load_inertial(rows[0]["inertial_path"], cfg["data"]["inertial"]["length"])
        shapes = {"depth_clip": list(d0.shape), "depth_dtype": str(d0.dtype),
                  "depth_range": [float(d0.min()), float(d0.max())],
                  "inertial": list(i0.shape), "inertial_dtype": str(i0.dtype)}
    except Exception as e:
        shapes = {"decode_error": repr(e)}

    stop_flags = []
    if audit["intersection"] != EXPECTED_USABLE:
        stop_flags.append(f"intersection {audit['intersection']} != expected {EXPECTED_USABLE}")
    if not disjoint:
        stop_flags.append(f"subject overlap between splits: {overlaps}")
    if set(audit["subjects_present"]) != set(range(1, 9)):
        stop_flags.append(f"subjects present != 1..8: {audit['subjects_present']}")
    if "decode_error" in shapes:
        stop_flags.append("sample decode failed")

    lines = []
    A = lines.append
    A("# UTD-MHAD data integrity audit\n")
    A(f"_Git `{git_hash()}`. Raw dir `{cfg['data']['raw_dir']}`. Modalities: depth + inertial (RGB/skeleton NOT used)._\n")
    A(f"**STOP flags:** {'NONE — audit clean' if not stop_flags else stop_flags}\n")
    A("## Trial counts")
    A(f"- depth trials: **{audit['depth_trials']}**")
    A(f"- inertial trials: **{audit['inertial_trials']}**")
    A(f"- **intersection (usable, both modalities): {audit['intersection']}** (expected {EXPECTED_USABLE})")
    A(f"- depth-only (missing inertial): {audit['n_depth_only']} {audit['depth_only'] if audit['n_depth_only'] else ''}")
    A(f"- inertial-only (missing depth): {audit['n_inertial_only']} {audit['inertial_only'] if audit['n_inertial_only'] else ''}")
    A(f"- assigned to a split: {audit['assigned']}\n")
    A("## Subjects & split (subject-disjoint)")
    A(f"- subjects present: {[int(s) for s in audit['subjects_present']]}")
    A(f"- TRAIN subjects {sorted(int(s) for s in subs['train'])}: {audit['per_split'].get('train')} clips")
    A(f"- VAL subjects {sorted(int(s) for s in subs['val'])}: {audit['per_split'].get('val')} clips")
    A(f"- TEST subjects {sorted(int(s) for s in subs['test'])}: {audit['per_split'].get('test')} clips")
    A(f"- overlaps: train∩val={overlaps['train_val']}, train∩test={overlaps['train_test']}, val∩test={overlaps['val_test']}")
    A(f"- **subject-disjoint: {disjoint}**\n")
    A("## Per-subject clip counts")
    for s in sorted(audit["per_subject"]):
        A(f"- subject {s}: {audit['per_subject'][s]}")
    A("\n## Per-action clip counts")
    ac = df.groupby("action").size()
    A(f"- actions present: {sorted(int(x) for x in df.action.unique())} (n={df.action.nunique()})")
    A(f"- min/median/max per action: {int(ac.min())}/{int(ac.median())}/{int(ac.max())}\n")
    A("## Sample decode (shape check)")
    for k, v in shapes.items():
        A(f"- {k}: {v}")
    (ROOT / "docs" / "audits" / "DATA_AUDIT_UTD_MHAD.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nSTOP_FLAGS:", stop_flags if stop_flags else "NONE")
    return 0 if not stop_flags else 1


if __name__ == "__main__":
    sys.exit(main())
