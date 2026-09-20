"""UTD-MHAD manifest: parse filenames, build depth/inertial trial-key intersection,
and apply the official subject-disjoint split (train {1,3,5} / val {7} / test {2,4,6,8}).

Filename convention (documented): a<i>_s<j>_t<k>_<modality>.mat  (action i, subject j, trial k).
Trial key = a<i>_s<j>_t<k>. A trial is USABLE only if present in BOTH depth and inertial.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

KEY_RE = re.compile(r"a(\d+)_s(\d+)_t(\d+)", re.IGNORECASE)


def _find_dir(raw_dir: Path, name: str) -> Path:
    """Locate the modality subdir (Depth/ or Inertial/) case-insensitively, else the raw root."""
    for p in raw_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return raw_dir


def scan_modality(raw_dir: Path, name: str) -> dict:
    """Return {trial_key: filepath} for one modality (files matched by name + a_s_t pattern)."""
    d = _find_dir(raw_dir, name)
    out = {}
    for fp in glob.glob(str(d / "**" / "*.mat"), recursive=True):
        base = os.path.basename(fp)
        if name.lower() not in base.lower():
            continue
        m = KEY_RE.search(base)
        if not m:
            continue
        i, j, k = (int(x) for x in m.groups())
        out[f"a{i}_s{j}_t{k}"] = fp
    return out


def subject_of(key: str) -> int:
    return int(KEY_RE.search(key).group(2))


def action_of(key: str) -> int:
    return int(KEY_RE.search(key).group(1))


def build_manifest(raw_dir, split):
    """Return (rows, audit). rows = list of {key, action, subject, split, depth_path, inertial_path}."""
    raw_dir = Path(raw_dir)
    depth = scan_modality(raw_dir, "Depth")
    iner = scan_modality(raw_dir, "Inertial")
    dk, ik = set(depth), set(iner)
    inter = sorted(dk & ik, key=lambda k: (action_of(k), subject_of(k), k))

    subj2split = {}
    for sp, subs in split.items():
        for s in subs:
            subj2split[int(s)] = sp

    rows = []
    for key in inter:
        s = subject_of(key)
        sp = subj2split.get(s)
        if sp is None:
            continue  # subject not in any split (should not happen for 1..8)
        rows.append({"key": key, "action": action_of(key), "subject": s, "split": sp,
                     "depth_path": depth[key], "inertial_path": iner[key]})

    audit = {
        "depth_trials": len(dk), "inertial_trials": len(ik),
        "intersection": len(inter),
        "depth_only": sorted(dk - ik), "inertial_only": sorted(ik - dk),
        "n_depth_only": len(dk - ik), "n_inertial_only": len(ik - dk),
        "subjects_present": sorted({subject_of(k) for k in (dk | ik)}),
        "per_split": {sp: sum(1 for r in rows if r["split"] == sp) for sp in split},
        "per_subject": {s: sum(1 for r in rows if r["subject"] == s)
                        for s in sorted({r["subject"] for r in rows})},
        "assigned": len(rows),
    }
    return rows, audit
