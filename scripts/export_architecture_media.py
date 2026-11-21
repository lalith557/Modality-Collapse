"""Thumbnails for the Model Architecture figure, produced by the REAL preprocessing code.

Nothing is drawn by hand: every image/curve is a model input (or the raw signal it came
from) computed by the training preprocessing on a stated example:

  CREMA-D audio  : samples/external/speech_tts.wav (Windows TTS, not CREMA-D) -> load_audio_logmel
  CREMA-D video  : samples/external/face_zoom.webm (public-domain photo, not CREMA-D) -> 8 grey 64x64 frames
  UTD-MHAD       : test-split trial a3_s4_t2 (depth .npy / inertial .csv converted from the dataset .mat)
                   -> load_depth_clip (16 x 32x32, min-max) and resample-to-128 + train z-score

Colour maps are display choices only (the tensors are single-channel). Writes
website/public/arch/*.png and website/src/data/architecture.media.json.
Usage: python scripts/export_architecture_media.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PIL import Image  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from data.crema_d import load_audio_logmel  # noqa: E402
from inference.cremad_preprocessor import load_video_frames_streaming  # noqa: E402
from inference.utd_preprocessor import prepare_depth, prepare_inertial  # noqa: E402

EXT = ROOT / "samples" / "external"
OUT_IMG = ROOT / "website" / "public" / "arch"
OUT_JSON = ROOT / "website" / "src" / "data" / "architecture.media.json"


def to_png(arr01: np.ndarray, path: Path, cmap: str | None = None, scale: int = 1) -> None:
    a = np.clip(arr01, 0, 1)
    if cmap:
        rgb = (matplotlib.colormaps[cmap](a)[..., :3] * 255).astype(np.uint8)
    else:
        rgb = (a * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path, optimize=True)


def minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def envelope(x: np.ndarray, n: int) -> list[float]:
    """Peak envelope of a waveform in n bins, normalised to [-1, 1] (for the raw-audio icon)."""
    x = x.astype(np.float64)
    x = x / (np.abs(x).max() or 1.0)
    bins = np.array_split(x, n)
    return [round(float(b[np.argmax(np.abs(b))]), 3) for b in bins]


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    wav = EXT / "speech_tts.wav"
    vid = EXT / "face_zoom.webm"
    depth = EXT / "utd_conv" / "a3_s4_t2_depth.npy"
    iner = EXT / "utd_conv" / "a3_s4_t2_inertial.csv"
    missing = [p for p in (wav, vid, depth, iner) if not p.exists()]
    if missing:
        sys.exit(f"missing {missing}; run python scripts/make_external_samples.py first")

    # CREMA-D audio: raw waveform envelope + the real log-mel tensor [1, 64, 188]
    sr, raw = wavfile.read(wav)
    raw = raw.mean(axis=1) if raw.ndim > 1 else raw
    logmel = load_audio_logmel(str(wav))[0].numpy()               # (64, 188), low freq at row 0
    to_png(minmax(logmel)[::-1], OUT_IMG / "cremad_logmel.png", "magma")

    # CREMA-D video: the 8 real 64x64 grey frames, side by side
    v = load_video_frames_streaming(vid)
    frames = v.tensor[:, 0].numpy()                               # (8, 64, 64) in [0, 1]
    to_png(np.concatenate(list(frames), axis=1), OUT_IMG / "cremad_frames.png")

    # UTD-MHAD depth: real min-max clip (16, 1, 32, 32); raw frames for the input icon
    stats = np.load(ROOT / "checkpoints" / "utd_mhad" / "baseline" / "normalization_stats.npz", allow_pickle=False)
    work = ROOT / "samples" / "_arch_tmp"
    work.mkdir(exist_ok=True)
    try:
        di = prepare_depth(depth, "npy", work)
        d = di.tensor[:, 0].numpy().astype(np.float32)              # (16, 32, 32) min-max, the model input
        to_png(np.concatenate(list(d), axis=1), OUT_IMG / "utd_depth.png", "viridis", scale=2)
        raw_depth = di.raw_thw                                      # (T, H, W) raw millimetres
        idx = np.linspace(0, raw_depth.shape[0] - 1, 3).round().astype(int)
        to_png(np.concatenate([minmax(raw_depth[i]) for i in idx], axis=1), OUT_IMG / "utd_depth_raw.png", "viridis")
        ii = prepare_inertial(iner, "csv", work, stats["inertial_channel_mean"], stats["inertial_channel_std"])
        z = ii.tensor.numpy().astype(np.float32)                    # (6, 128) train z-scored, the model input
        raw_i = ii.raw_t6                                           # (T, 6) raw units
    finally:
        for f in work.glob("*"):
            f.unlink()
        work.rmdir()
    media = {
        "generated_by": "scripts/export_architecture_media.py",
        "note": "Real preprocessing outputs on stated examples; colour maps are display choices.",
        "sources": {
            "cremad_audio": "samples/external/speech_tts.wav — Windows text-to-speech (not CREMA-D) through data.crema_d.load_audio_logmel",
            "cremad_video": "samples/external/face_zoom.webm — public-domain photograph (not CREMA-D) through the training frame sampler",
            "utd": "UTD-MHAD test-split trial a3_s4_t2 through data.utd_mhad.load_depth_clip / load_inertial + train z-score",
        },
        "audio_envelope": envelope(raw, 72),
        "audio_rate": int(sr),
        "logmel_shape": list(logmel.shape),
        "frames_shape": list(frames.shape),
        "depth_shape": list(d.shape),
        "inertial_raw": [[round(float(x), 3) for x in raw_i[:: max(1, len(raw_i) // 64), c]] for c in range(6)],
        "inertial_z": [[round(float(x), 3) for x in z[c]] for c in range(6)],
    }
    OUT_JSON.write_text(json.dumps(media, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in sorted(OUT_IMG.glob("*.png")):
        print(p.relative_to(ROOT), Image.open(p).size, p.stat().st_size, "bytes")
    print("wrote", OUT_JSON.relative_to(ROOT))


if __name__ == "__main__":
    main()
