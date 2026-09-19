"""CREMA-D media integrity validation (Phase 5-6). Categorizes every file, checks
signatures, decodes a stratified sample, and verifies manifest/actor/label
consistency. Writes results/cremad/cremad_validation.json. Does NOT train."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.crema_d import load_audio_logmel, load_video_frames  # noqa: E402
import cv2  # noqa: E402
from scipy.io import wavfile  # noqa: E402


def categorize(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    sz = p.stat().st_size
    if sz == 0:
        return "zero"
    if sz <= 200:
        try:
            head = p.read_bytes()[:40]
            if head.startswith(b"version https://git-lfs"):
                return "lfs_stub"
        except OSError:
            pass
        return "tiny"
    return "present"


def main():
    man = pd.read_csv(ROOT / "data" / "cremad" / "cremad_manifest.csv")
    rep = {"n_clips": int(len(man)), "n_actors": int(man.actor_id.nunique()),
           "n_emotions": int(man.emotion.nunique()), "n_sentences": int(man.sentence.nunique())}

    # 1) categorize every file
    acat = man.audio_path.map(categorize).value_counts().to_dict()
    vcat = man.video_path.map(categorize).value_counts().to_dict()
    rep["audio_categories"] = {k: int(v) for k, v in acat.items()}
    rep["video_categories"] = {k: int(v) for k, v in vcat.items()}

    # 2) duplicates (by clip_id) and extra files on disk
    rep["duplicate_clip_ids"] = int(man.clip_id.duplicated().sum())
    disk_wavs = {p.stem for p in (ROOT.parent / "CREMA-D-full" / "AudioWAV").glob("*.wav")}
    rep["disk_audio_files"] = len(disk_wavs)
    rep["manifest_minus_disk"] = len(set(man.clip_id) - disk_wavs)
    rep["disk_minus_manifest"] = len(disk_wavs - set(man.clip_id))

    # 3) signature + decode on a stratified sample (per emotion x split)
    rng = np.random.default_rng(0)
    sample = man.groupby(["emotion", "split"], group_keys=False).apply(
        lambda g: g.sample(min(4, len(g)), random_state=0)).reset_index(drop=True)
    a_ok = v_ok = a_sig = v_sig = 0
    a_dur, v_frames, a_sr = [], [], []
    fails = []
    for _, r in sample.iterrows():
        # signatures
        try:
            with open(r.audio_path, "rb") as f:
                a_sig += 1 if f.read(4) == b"RIFF" else 0
            with open(r.video_path, "rb") as f:
                v_sig += 1 if f.read(3) == b"FLV" else 0
        except OSError:
            pass
        # audio decode + duration
        try:
            sr, data = wavfile.read(r.audio_path)
            a_dur.append(len(data) / sr); a_sr.append(int(sr))
            _ = load_audio_logmel(r.audio_path)
            a_ok += 1
        except Exception as e:
            fails.append(("audio", r.clip_id, str(e)[:60]))
        # video decode + frame count
        try:
            cap = cv2.VideoCapture(str(r.video_path)); n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            cap.release()
            v_frames.append(n)
            _ = load_video_frames(r.video_path)
            v_ok += 1 if n > 0 else 0
        except Exception as e:
            fails.append(("video", r.clip_id, str(e)[:60]))

    rep["sample_size"] = int(len(sample))
    rep["audio_riff_signature_ok"] = int(a_sig)
    rep["video_flv_signature_ok"] = int(v_sig)
    rep["audio_decoded_ok"] = int(a_ok)
    rep["video_decoded_ok"] = int(v_ok)
    rep["audio_duration_s"] = {"min": round(float(np.min(a_dur)), 2), "max": round(float(np.max(a_dur)), 2),
                               "mean": round(float(np.mean(a_dur)), 2)} if a_dur else {}
    rep["audio_sample_rates"] = sorted(set(a_sr))
    rep["video_frame_count"] = {"min": int(np.min(v_frames)), "max": int(np.max(v_frames)),
                                "mean": round(float(np.mean(v_frames)), 1)} if v_frames else {}
    rep["decode_failures"] = fails

    out = ROOT / "results" / "cremad" / "cremad_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)

    print(json.dumps(rep, indent=2))
    identity_ok = (rep["n_clips"] == 7442 and rep["n_actors"] == 91 and rep["n_emotions"] == 6
                   and rep["n_sentences"] == 12)
    media_ok = (rep["audio_categories"].get("present", 0) == 7442
                and rep["video_categories"].get("present", 0) == 7442
                and not fails and rep["manifest_minus_disk"] == 0)
    print("\nIDENTITY_OK:", identity_ok, "| MEDIA_OK:", media_ok)


if __name__ == "__main__":
    main()
