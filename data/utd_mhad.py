"""UTD-MHAD encoder-input loaders (depth + inertial), deterministic preprocessing.

Per PREREGISTRATION_UTD_MHAD_SCALE.md:
  depth    -> uniform T frames, resized HxH, per-clip min-max to [0,1] -> [T,1,H,H]
  inertial -> linear-resample time to L, 6 channels (accel+gyro)       -> [6,L] (raw;
              per-channel z-score with TRAIN stats is applied in experiments/utd_extract.py)

.mat variables are auto-detected (single non-meta key) and re-oriented by shape so the
code is robust to (H,W,T)/(T,H,W) and (T,6)/(6,T) conventions.
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False


def _main_array(path):
    """Return the single data array from a .mat (ignoring __header__/__version__/__globals__)."""
    m = sio.loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    if not keys:
        raise ValueError(f"no data variable in {path}")
    # pick the largest array (UTD files have exactly one payload variable)
    k = max(keys, key=lambda kk: np.asarray(m[kk]).size)
    return np.asarray(m[k]), k


def _resize_frame(f, hw):
    f = f.astype(np.float32)
    if _HAVE_CV2:
        return cv2.resize(f, (hw, hw), interpolation=cv2.INTER_AREA)
    # numpy fallback: nearest-ish via index sampling
    yi = np.linspace(0, f.shape[0] - 1, hw).round().astype(int)
    xi = np.linspace(0, f.shape[1] - 1, hw).round().astype(int)
    return f[yi][:, xi]


def load_depth_clip(path, n_frames=16, hw=32):
    """-> float16 [n_frames, 1, hw, hw] in [0,1]."""
    arr, _ = _main_array(path)
    if arr.ndim != 3:
        raise ValueError(f"depth expected 3D, got {arr.shape} in {path}")
    # orient to (T, H, W): time axis = the one that is NOT a plausible spatial dim.
    # UTD depth is (240,320,T). If last axis looks spatial and axis0 doesn't, treat axis0 as T.
    s = arr.shape
    if s[2] <= s[0] and s[2] <= s[1]:
        t_axis = 2                      # (H, W, T) UTD standard
    elif s[0] <= s[1] and s[0] <= s[2]:
        t_axis = 0                      # (T, H, W)
    else:
        t_axis = int(np.argmin(s))
    arr = np.moveaxis(arr, t_axis, 0)   # (T, H, W)
    T = arr.shape[0]
    idx = np.linspace(0, T - 1, n_frames).round().astype(int)
    frames = np.stack([_resize_frame(arr[i], hw) for i in idx], axis=0)  # (n_frames, hw, hw)
    lo, hi = float(frames.min()), float(frames.max())
    frames = (frames - lo) / (hi - lo) if hi > lo else np.zeros_like(frames)
    return frames[:, None, :, :].astype(np.float16)                       # (T,1,hw,hw)


def load_inertial(path, length=128):
    """-> float32 [6, length] (raw, pre-standardization)."""
    arr, _ = _main_array(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] == 6:               # (T, 6)
        sig = arr
    elif arr.shape[0] == 6:             # (6, T)
        sig = arr.T
    else:                               # fall back: assume longer axis is time
        sig = arr if arr.shape[0] >= arr.shape[1] else arr.T
        sig = sig[:, :6] if sig.shape[1] >= 6 else np.pad(sig, ((0, 0), (0, 6 - sig.shape[1])))
    T = sig.shape[0]
    xnew = np.linspace(0, T - 1, length)
    xold = np.arange(T)
    out = np.stack([np.interp(xnew, xold, sig[:, c]) for c in range(6)], axis=0)  # (6, length)
    return out.astype(np.float32)
