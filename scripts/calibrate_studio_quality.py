"""Calibrate the Studio's CREMA-D quality/domain thresholds on VALIDATION media.

Thresholds that describe "unlike the training distribution" are set from the validation
split (1st/99th percentiles); a few physically-motivated ones are fixed constants (e.g.
the 64 px model input size). TEST media is processed only to REPORT how often each check
would wrongly fire on in-distribution data. Writes
checkpoints/cremad/baseline/quality_reference.json and refreshes the package file hashes.

Usage: python scripts/calibrate_studio_quality.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from inference.cremad_preprocessor import load_video_frames_streaming, prepare_audio
from inference.quality_checks import assess_audio, assess_video, audio_metrics, video_metrics

PKG = ROOT / "checkpoints" / "cremad" / "baseline"


def metrics_for(df: pd.DataFrame, tmp: Path, label: str) -> pd.DataFrame:
    rows, t0 = [], time.time()
    for i, r in df.reset_index(drop=True).iterrows():
        a = prepare_audio(Path(r.audio_path), "wav", tmp)
        am = audio_metrics(a.mono_native, a.native_rate)
        v = load_video_frames_streaming(r.video_path)
        vm = video_metrics(v.selected_bgr, v.tensor, v.width, v.height, v.n_frames, v.fps, v.duration_s)
        rows.append({"clip_id": r.clip_id, **{f"a_{k}": x for k, x in am.items()}, **{f"v_{k}": x for k, x in vm.items()}})
        if (i + 1) % 250 == 0:
            print(f"  {label} {i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    man = pd.read_csv(ROOT / "data/cremad/cremad_manifest.csv")
    ok = man[(man.audio_status == "present") & (man.video_status == "present")]
    tmp = Path(tempfile.mkdtemp())
    val = metrics_for(ok[ok.split == "val"], tmp, "val")
    test = metrics_for(ok[ok.split == "test"], tmp, "test")

    q = lambda col, p: float(np.percentile(val[col], p))
    audio_t = {"silence_rms_dbfs": -60.0, "min_duration_s": 0.5,
               "min_active_fraction": q("a_active_fraction", 1), "min_modulation_ratio": q("a_modulation_ratio", 1),
               "max_clip_fraction": max(q("a_clip_fraction", 99), 1e-3), "min_dynamic_range_db": q("a_dynamic_range_db", 1)}
    video_t = {"min_face_fraction": min(0.25, q("v_face_fraction", 1)), "min_side_px": 64,
               "min_brightness": min(0.08, q("v_mean_brightness", 0.5)), "max_brightness": max(0.92, q("v_mean_brightness", 99.5)),
               "min_blur_laplacian_var": q("v_blur_laplacian_var", 1), "min_duration_s": 0.5,
               "min_frame_motion": q("v_frame_motion", 1), "long_clip_s": 6.0}
    ref = {"calibrated_on": "CREMA-D validation split (738 clips)", "audio_thresholds": audio_t, "video_thresholds": video_t}

    def rates(df):
        out = {}
        for name, fn, prefix in (("audio", assess_audio, "a_"), ("video", assess_video, "v_")):
            sev = {"unsupported": 0, "low_quality": 0}; codes = {}
            for _, r in df.iterrows():
                m = {k[len(prefix):]: r[k] for k in df.columns if k.startswith(prefix)}
                iss = fn(m, ref)
                for s in sev:
                    sev[s] += any(x["severity"] == s for x in iss)
                for x in iss:
                    codes[x["code"]] = codes.get(x["code"], 0) + 1
            out[name] = {"n": len(df), "unsupported_rate": sev["unsupported"] / len(df),
                         "low_quality_rate": sev["low_quality"] / len(df),
                         "per_check_rate": {k: v / len(df) for k, v in sorted(codes.items())}}
        return out

    ref["false_flag_rates"] = {"validation": rates(val), "test_evaluation_only": rates(test)}
    ref["validation_metric_summary"] = {c: {"p1": q(c, 1), "p50": q(c, 50), "p99": q(c, 99)}
                                        for c in val.columns if c != "clip_id"}
    (PKG / "quality_reference.json").write_text(json.dumps(ref, indent=2))
    meta = json.loads((PKG / "metadata.json").read_text())
    meta["package_files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(PKG.iterdir()) if p.name != "metadata.json"}
    (PKG / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"thresholds": {"audio": audio_t, "video": video_t},
                      "false_flag_rates": ref["false_flag_rates"]}, indent=1))


if __name__ == "__main__":
    main()
