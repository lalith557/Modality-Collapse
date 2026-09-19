"""Build the CREMA-D manifest + actor-independent split (media-independent).

Parses the real filenames (ActorID_Sentence_Emotion_Level) and VideoDemographics.csv;
assigns a frozen, reproducible, ACTOR-DISJOINT train/val/test split (stratified by
sex). Records media paths and a preprocessing_status. Does NOT require the media
files themselves (which are currently git-LFS pointer stubs). Run once; commit the
manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_BASE = Path(r"C:/Users/Lalith Gona/HTML/Deep Learning Project 2")


def _pick_media_root() -> Path:
    """Prefer the LFS-pulled full clone if its media are real; else the stub copy."""
    full = _BASE / "CREMA-D-full"
    probe = full / "AudioWAV" / "1001_DFA_ANG_XX.wav"
    if probe.exists() and probe.stat().st_size > 1024:
        return full
    return _BASE / "CREMA-D-master"


CREMAD = _pick_media_root()
EMO = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
SPLIT_SEED = 42
FRAC = {"train": 0.70, "val": 0.10, "test": 0.20}


def media_status(path: Path) -> str:
    try:
        sz = path.stat().st_size
    except OSError:
        return "missing"
    return "present" if sz > 1024 else "lfs_stub"      # 130-byte pointers => stub


def actor_split(actor_ids, demo):
    """Actor-disjoint split, stratified by sex, deterministic."""
    rng = np.random.default_rng(SPLIT_SEED)
    sex = demo.set_index("ActorID")["Sex"].to_dict()
    assign = {}
    for s in ("Male", "Female"):
        actors = sorted(a for a in actor_ids if sex.get(int(a)) == s)
        rng.shuffle(actors)
        n = len(actors)
        n_tr = int(round(FRAC["train"] * n)); n_va = int(round(FRAC["val"] * n))
        for i, a in enumerate(actors):
            assign[a] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    # any actor with unknown sex -> deterministic fallback
    for a in actor_ids:
        assign.setdefault(a, "train")
    return assign


def main():
    wavs = sorted(CREMAD.glob("AudioWAV/*.wav"))
    if not wavs:
        print("No AudioWAV files found.", file=sys.stderr); sys.exit(1)
    demo = pd.read_csv(CREMAD / "VideoDemographics.csv")

    rows = []
    for w in wavs:
        actor, sent, emo, lvl = w.stem.split("_")
        vid = CREMAD / "VideoFlash" / f"{w.stem}.flv"
        rows.append({
            "clip_id": w.stem, "actor_id": actor, "sentence": sent,
            "emotion": emo, "label": EMO.get(emo, -1), "level": lvl,
            "audio_path": str(w), "video_path": str(vid),
            "audio_status": media_status(w), "video_status": media_status(vid),
        })
    df = pd.DataFrame(rows)
    assign = actor_split(sorted(df.actor_id.unique()), demo)
    df["split"] = df.actor_id.map(assign)
    df["preprocessing_status"] = "media_absent" if (df.audio_status.eq("lfs_stub").all()) else "pending"

    # leakage assertion: actor-disjoint splits
    overlap = (df.groupby("actor_id")["split"].nunique() > 1).sum()
    assert overlap == 0, f"ACTOR LEAKAGE: {overlap} actors span multiple splits"

    for out in (ROOT / "data" / "cremad" / "cremad_manifest.csv",
                ROOT / "results" / "cremad" / "cremad_manifest.csv"):
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    print(f"clips={len(df)} actors={df.actor_id.nunique()} "
          f"media_present={df.audio_status.eq('present').sum()} (audio) / {df.video_status.eq('present').sum()} (video)")
    print("split actor counts:", df.groupby("split").actor_id.nunique().to_dict())
    print("split clip counts:", df.groupby("split").size().to_dict())
    print("actor-disjoint (0 overlap):", overlap == 0)
    print("emotion x split:\n", pd.crosstab(df.emotion, df.split))


if __name__ == "__main__":
    main()
