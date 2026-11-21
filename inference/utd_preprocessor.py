"""UTD-MHAD preprocessing for external inputs -- training-identical.

Accepted inputs (structured sensor data only; an RGB video is never treated as depth or
inertial data):
  depth    : UTD-style MATLAB v5 .mat (largest variable, e.g. d_depth (H,W,T) uint16 mm)
             or .npy array (T,H,W) / (H,W,T)
  inertial : UTD-style .mat (d_iner, T x 6) or .csv with 6 numeric columns
             [acc_x, acc_y, acc_z (g), gyro_x, gyro_y, gyro_z (deg/s)], optional header

.npy and .csv uploads are written to a temporary .mat and then read by the UNCHANGED
training functions data.utd_mhad.load_depth_clip / load_inertial, so every upload goes
through one code path. Inertial signals are then z-scored with the TRAIN statistics and,
like the training cache, rounded through float16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from data.utd_mhad import _main_array, load_depth_clip, load_inertial
from inference.input_validator import InputError

N_FRAMES, HW, LENGTH = 16, 32, 128
MAX_DEPTH_FRAMES = 2000
MAX_INERTIAL_SAMPLES = 20000


@dataclass
class DepthInput:
    tensor: torch.Tensor                     # [16,1,32,32]
    raw_thw: np.ndarray = field(repr=False)  # oriented raw array for QC
    raw_shape: tuple = ()
    source_kind: str = ""


@dataclass
class InertialInput:
    tensor: torch.Tensor                     # [6,128] z-scored
    raw_t6: np.ndarray = field(repr=False)
    resampled_raw: np.ndarray = field(repr=False)   # [6,128] before z-score
    n_samples: int = 0
    source_kind: str = ""


def _orient_thw(arr: np.ndarray) -> np.ndarray:
    s = arr.shape
    if s[2] <= s[0] and s[2] <= s[1]:
        t_axis = 2
    elif s[0] <= s[1] and s[0] <= s[2]:
        t_axis = 0
    else:
        t_axis = int(np.argmin(s))
    return np.moveaxis(arr, t_axis, 0)


def prepare_depth(src: Path, kind: str, workdir: Path) -> DepthInput:
    try:
        if kind == "npy":
            arr = np.load(str(src), allow_pickle=False)
            mat = workdir / "depth_from_npy.mat"
            sio.savemat(str(mat), {"d_depth": np.asarray(arr)})
            src = mat
        else:
            arr, _ = _main_array(str(src))
    except InputError:
        raise
    except Exception as e:  # noqa: BLE001
        raise InputError("depth_decode_failed", f"The depth file could not be read ({type(e).__name__}).", 422, "depth")
    arr = np.asarray(arr)
    if arr.ndim != 3 or not np.issubdtype(arr.dtype, np.number):
        raise InputError("depth_bad_shape", f"Depth must be a numeric 3-D array (frames x height x width); "
                         f"got shape {arr.shape}, dtype {arr.dtype}.", 422, "depth")
    thw = _orient_thw(arr)
    if thw.shape[0] > MAX_DEPTH_FRAMES or min(thw.shape[1:]) < 8:
        raise InputError("depth_bad_shape", f"Depth array {arr.shape} is implausible for a depth clip "
                         "(UTD-MHAD is 240 x 320 x ~40-125 frames).", 422, "depth")
    try:
        clip = load_depth_clip(str(src), N_FRAMES, HW)          # unchanged training function
    except Exception as e:  # noqa: BLE001
        raise InputError("depth_decode_failed", f"Training depth preprocessing failed ({type(e).__name__}).", 422, "depth")
    return DepthInput(tensor=torch.from_numpy(clip).float(), raw_thw=thw.astype(np.float32),
                      raw_shape=tuple(arr.shape), source_kind=kind)


def _parse_csv(src: Path) -> np.ndarray:
    text = src.read_text(encoding="utf-8", errors="strict")
    rows = [r for r in text.replace("\r", "").split("\n") if r.strip()]
    delim = next((d for d in (",", ";", "\t") if d in rows[0]), None)
    out = []
    for i, r in enumerate(rows):
        parts = [p.strip() for p in (r.split(delim) if delim else r.split())]
        try:
            out.append([float(p) for p in parts])
        except ValueError:
            if i == 0:
                continue          # header row
            raise InputError("inertial_bad_csv", f"Row {i+1} of the inertial CSV is not numeric.", 422, "inertial")
    if not out:
        raise InputError("inertial_bad_csv", "The inertial CSV has no numeric rows.", 422, "inertial")
    widths = {len(r) for r in out}
    if len(widths) != 1:
        raise InputError("inertial_bad_csv", "Inertial CSV rows have different numbers of columns.", 422, "inertial")
    return np.asarray(out, dtype=np.float64)


def prepare_inertial(src: Path, kind: str, workdir: Path, mean: np.ndarray, std: np.ndarray) -> InertialInput:
    try:
        if kind == "csv":
            sig = _parse_csv(src)
        else:
            sig, _ = _main_array(str(src))
            sig = np.asarray(sig, dtype=np.float64)
    except InputError:
        raise
    except Exception as e:  # noqa: BLE001
        raise InputError("inertial_decode_failed", f"The inertial file could not be read ({type(e).__name__}).", 422, "inertial")
    if sig.ndim != 2 or 6 not in sig.shape:
        raise InputError("inertial_bad_shape", f"Inertial data must be a T x 6 array (3 accelerometer + 3 "
                         f"gyroscope channels); got shape {sig.shape}.", 422, "inertial")
    t6 = sig if sig.shape[1] == 6 else sig.T
    if t6.shape[0] < 2 or t6.shape[0] > MAX_INERTIAL_SAMPLES:
        raise InputError("inertial_bad_shape", f"Inertial signal has {t6.shape[0]} samples; expected roughly "
                         "100-300 (UTD-MHAD, 50 Hz).", 422, "inertial")
    mat = workdir / "inertial_input.mat"
    sio.savemat(str(mat), {"d_iner": t6})
    try:
        res = load_inertial(str(mat), LENGTH)                 # unchanged training function -> [6,128]
    except Exception as e:  # noqa: BLE001
        raise InputError("inertial_decode_failed", f"Training inertial preprocessing failed ({type(e).__name__}).", 422, "inertial")
    z = ((res - mean[:, None]) / std[:, None]).astype(np.float16)    # train-stat z-score, cache dtype
    return InertialInput(tensor=torch.from_numpy(z).float(), raw_t6=t6.astype(np.float32),
                         resampled_raw=res, n_samples=int(t6.shape[0]), source_kind=kind)
