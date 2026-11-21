"""Input quality and domain checks.

Metric functions are pure and are used twice: by scripts/calibrate_studio_quality.py on
CREMA-D *validation* media (to set data-driven thresholds, saved as quality_reference.json
in the checkpoint package) and at inference time. TEST media is only used to *report* the
resulting false-reject rates.

Severity levels
  unsupported : the input is outside the task domain as far as these checks can tell
                (no speech-like activity; no face inside the model's field of view;
                malformed sensor data). No prediction is returned.
  low_quality : a prediction is returned, but a quality problem makes it less trustworthy.
  info        : a neutral fact about how the input was used (e.g. truncation to 3.0 s).

Heuristic limits (stated, not hidden): the speech checks use frame energy and syllabic
modulation; they reject silence, steady tones and steady noise, but they cannot reliably
tell speech from singing or from rhythmic music. The face check is an OpenCV Haar cascade
for *frontal* faces, run on the centre crop the model actually sees.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torchaudio

ANALYSIS_SR = 16000
WINDOW_S = 3.0
_FACE = None


def _face_cascade():
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    return _FACE


# ---------------------------------------------------------------- audio
def audio_metrics(mono_native: np.ndarray, native_rate: int) -> dict:
    x = torch.from_numpy(np.asarray(mono_native, dtype=np.float32))[None, :]
    if native_rate != ANALYSIS_SR:
        x = torchaudio.functional.resample(x, native_rate, ANALYSIS_SR)
    full = x[0].numpy()
    duration = len(mono_native) / float(native_rate)
    w = full[: int(WINDOW_S * ANALYSIS_SR)]               # the window the model sees
    eps = 1e-12
    rms = float(np.sqrt(np.mean(w ** 2) + eps)) if w.size else 0.0
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    clip_frac = float(np.mean(np.abs(w) >= 0.999)) if w.size else 0.0
    # 25 ms frames / 10 ms hop; speech-band (300-3400 Hz) energy
    nfft, hop = 400, 160
    if w.size < nfft:
        return {"duration_s": duration, "window_s": w.size / ANALYSIS_SR, "rms_dbfs": 20 * np.log10(rms + eps),
                "peak": peak, "clip_fraction": clip_frac, "dynamic_range_db": 0.0,
                "active_fraction": 0.0, "modulation_ratio": 0.0}
    spec = torch.stft(torch.from_numpy(w), nfft, hop, window=torch.hann_window(nfft), return_complex=True).abs() ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / ANALYSIS_SR)
    band = spec[(freqs >= 300) & (freqs <= 3400)].sum(0).numpy() / nfft
    band_db = 10 * np.log10(band + eps)
    floor = np.percentile(band_db, 10)
    active = (band_db > floor + 10.0) & (band_db > -60.0)
    env = band - band.mean()
    mf = np.fft.rfftfreq(len(env), hop / ANALYSIS_SR)
    mp = np.abs(np.fft.rfft(env)) ** 2
    denom = mp[(mf >= 0.5) & (mf <= 25)].sum()
    mod = float(mp[(mf >= 2) & (mf <= 8)].sum() / denom) if denom > 0 else 0.0
    return {"duration_s": float(duration), "window_s": float(min(duration, WINDOW_S)),
            "rms_dbfs": float(20 * np.log10(rms + eps)), "peak": peak, "clip_fraction": clip_frac,
            "dynamic_range_db": float(np.percentile(band_db, 95) - floor),
            "active_fraction": float(active.mean()), "modulation_ratio": mod}


def assess_audio(m: dict, ref: dict) -> list[dict]:
    t = ref["audio_thresholds"]
    issues = []
    if m["peak"] < 1e-4 or m["rms_dbfs"] < t["silence_rms_dbfs"]:
        issues.append({"code": "silent_audio", "severity": "unsupported",
                       "message": f"The audio is silent or nearly silent (RMS {m['rms_dbfs']:.0f} dBFS)."})
        return issues
    if m["active_fraction"] < t["min_active_fraction"] or m["modulation_ratio"] < t["min_modulation_ratio"]:
        issues.append({"code": "no_speech_detected", "severity": "unsupported",
                       "message": "No speech-like activity was found in the first 3.0 s (energy and "
                                  "syllabic-modulation heuristic). The model only handles spoken "
                                  "emotional utterances; music, tones or environmental sound are out of scope."})
    if m["duration_s"] < t["min_duration_s"]:
        issues.append({"code": "audio_too_short", "severity": "low_quality",
                       "message": f"The clip is {m['duration_s']:.2f} s; it is zero-padded to 3.0 s, so most "
                                  "of the model's input window is silence."})
    if m["clip_fraction"] > t["max_clip_fraction"]:
        issues.append({"code": "audio_clipping", "severity": "low_quality",
                       "message": f"{100*m['clip_fraction']:.2f}% of samples are clipped."})
    if m["dynamic_range_db"] < t["min_dynamic_range_db"]:
        issues.append({"code": "audio_noisy", "severity": "low_quality",
                       "message": f"Low speech-to-background range ({m['dynamic_range_db']:.1f} dB); "
                                  "the recording may be noisy."})
    if m["duration_s"] > WINDOW_S + 0.05:
        issues.append({"code": "audio_truncated", "severity": "info",
                       "message": f"Only the first 3.0 s of {m['duration_s']:.1f} s are used, exactly as in training."})
    return issues


# ---------------------------------------------------------------- video
def _centre_square(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    s = min(h, w)
    return gray[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def video_metrics(selected_bgr: list, tensor: torch.Tensor, width: int, height: int,
                  n_frames: int, fps: float, duration_s: float) -> dict:
    blur, bright, faces = [], [], 0
    for f in selected_bgr:
        g = _centre_square(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        g256 = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
        blur.append(float(cv2.Laplacian(g256, cv2.CV_64F).var()))
        bright.append(float(g.mean() / 255.0))
        det = _face_cascade().detectMultiScale(g256, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        faces += int(len(det) > 0)
    fr = tensor[:, 0].numpy()
    motion = float(np.mean(np.abs(np.diff(fr, axis=0)))) if len(fr) > 1 else 0.0
    return {"width": int(width), "height": int(height), "min_side_px": int(min(width, height)),
            "n_frames": int(n_frames), "fps": float(fps), "duration_s": float(duration_s),
            "mean_brightness": float(np.mean(bright)), "blur_laplacian_var": float(np.median(blur)),
            "face_fraction": faces / max(1, len(selected_bgr)), "frame_motion": motion}


def assess_video(m: dict, ref: dict) -> list[dict]:
    t = ref["video_thresholds"]
    issues = []
    if m["face_fraction"] < t["min_face_fraction"]:
        issues.append({"code": "no_face_detected", "severity": "unsupported",
                       "message": f"A frontal face was found in {round(m['face_fraction']*8)}/8 sampled frames "
                                  "inside the centre crop the model sees. The model was trained only on "
                                  "close-up frontal faces of speakers."})
    if m["min_side_px"] < t["min_side_px"]:
        issues.append({"code": "video_low_resolution", "severity": "low_quality",
                       "message": f"The video is {m['width']}x{m['height']}; frames are smaller than the "
                                  f"64x64 model input and must be upsampled."})
    if m["mean_brightness"] < t["min_brightness"]:
        issues.append({"code": "video_too_dark", "severity": "low_quality",
                       "message": f"Frames are very dark (mean brightness {m['mean_brightness']:.2f})."})
    if m["mean_brightness"] > t["max_brightness"]:
        issues.append({"code": "video_overexposed", "severity": "low_quality",
                       "message": f"Frames are very bright (mean brightness {m['mean_brightness']:.2f})."})
    if m["blur_laplacian_var"] < t["min_blur_laplacian_var"]:
        issues.append({"code": "video_blurry", "severity": "low_quality",
                       "message": "Frames are blurrier than nearly all training-distribution clips."})
    if m["n_frames"] < 8:
        issues.append({"code": "video_few_frames", "severity": "low_quality",
                       "message": f"Only {m['n_frames']} frames are available, so some of the 8 model frames repeat."})
    if m["duration_s"] and m["duration_s"] < t["min_duration_s"]:
        issues.append({"code": "video_too_short", "severity": "low_quality",
                       "message": f"The video is {m['duration_s']:.2f} s long."})
    if m["frame_motion"] < t["min_frame_motion"]:
        issues.append({"code": "video_static", "severity": "low_quality",
                       "message": "The sampled frames are nearly identical (a still image or frozen video); "
                                  "training clips contain a moving, speaking face."})
    if m["duration_s"] > t["long_clip_s"]:
        issues.append({"code": "video_long", "severity": "info",
                       "message": f"The 8 frames are spread over {m['duration_s']:.1f} s; training clips are "
                                  "about 1.5-5 s, so they are sampled more sparsely in time."})
    return issues


# ---------------------------------------------------------------- UTD-MHAD
def assess_depth(arr_thw: np.ndarray) -> list[dict]:
    issues = []
    if float(arr_thw.max()) <= 0 or float(arr_thw.max()) == float(arr_thw.min()):
        issues.append({"code": "depth_empty", "severity": "unsupported",
                       "message": "The depth array is constant or all zero."})
        return issues
    if arr_thw.shape[0] < 16:
        issues.append({"code": "depth_few_frames", "severity": "low_quality",
                       "message": f"Only {arr_thw.shape[0]} depth frames; some of the 16 model frames repeat."})
    if float(arr_thw.max()) > 20000 or float(arr_thw.max()) < 50:
        issues.append({"code": "depth_units", "severity": "low_quality",
                       "message": f"Depth values span 0-{float(arr_thw.max()):.0f}; UTD-MHAD Kinect depth is in "
                                  "millimetres (about 0-4000). Values in other units change the representation."})
    return issues


def assess_inertial(sig_t6: np.ndarray) -> list[dict]:
    issues = []
    if sig_t6.shape[0] < 32:
        issues.append({"code": "inertial_too_short", "severity": "low_quality",
                       "message": f"Only {sig_t6.shape[0]} inertial samples; UTD-MHAD trials have roughly 100-300."})
    acc_mag = np.linalg.norm(sig_t6[:, :3], axis=1)
    if np.median(acc_mag) > 4.0:
        issues.append({"code": "inertial_units_accel", "severity": "low_quality",
                       "message": f"Median accelerometer magnitude is {np.median(acc_mag):.1f}; UTD-MHAD uses g "
                                  "(about 1 at rest). Values near 9.8 suggest m/s^2."})
    if np.abs(sig_t6[:, 3:]).max() < 5.0 and np.abs(sig_t6[:, 3:]).max() > 0:
        issues.append({"code": "inertial_units_gyro", "severity": "low_quality",
                       "message": "Gyroscope values are all below 5; UTD-MHAD uses degrees/second. "
                                  "Values in rad/s would be about 57x too small."})
    if not np.isfinite(sig_t6).all():
        issues.append({"code": "inertial_nonfinite", "severity": "unsupported",
                       "message": "The inertial signal contains NaN or infinite values."})
    return issues
