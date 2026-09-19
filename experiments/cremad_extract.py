"""One-time CREMA-D feature extraction / caching (deterministic; does NOT change
the science -- decode is fixed and identical across all runs). Caches per split:
audio log-mel [N,1,64,188], video frames [N,8,1,64,64] (float16), labels, actors.
Records any decode failures. Run once; R1/R2/R3 then train on the cache.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.crema_d import load_audio_logmel, load_video_frames


def main():
    man = pd.read_csv(ROOT / "data" / "cremad" / "cremad_manifest.csv")
    present = man[(man.audio_status == "present") & (man.video_status == "present")]
    cache = ROOT / "data" / "cremad" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    failures = []
    t0 = time.time()
    for split in ("train", "val", "test"):
        df = present[present.split == split].reset_index(drop=True)
        A, V, Y, ACT = [], [], [], []
        for i, r in df.iterrows():
            try:
                a = load_audio_logmel(r.audio_path).numpy().astype(np.float16)
                v = load_video_frames(r.video_path).numpy().astype(np.float16)
            except Exception as e:
                failures.append({"clip_id": r.clip_id, "split": split, "err": str(e)[:80]})
                continue
            A.append(a); V.append(v); Y.append(int(r.label)); ACT.append(str(r.actor_id))
            if (i + 1) % 500 == 0:
                print(f"  {split} {i+1}/{len(df)}  ({time.time()-t0:.0f}s)", flush=True)
        np.save(cache / f"{split}_audio.npy", np.stack(A))
        np.save(cache / f"{split}_video.npy", np.stack(V))
        np.save(cache / f"{split}_y.npy", np.array(Y, dtype=np.int64))
        np.save(cache / f"{split}_actor.npy", np.array(ACT))
        print(f"[{split}] cached {len(A)} clips "
              f"(audio {np.stack(A).shape}, video {np.stack(V).shape})", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(cache / "extract_failures.csv", index=False)
    print(f"DONE. failures={len(failures)}  runtime={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
