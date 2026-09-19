"""CREMA-D real audio/video loaders + dataset (per PREREGISTRATION_CREMAD.md).

Audio  : WAV (scipy) -> 16 kHz mono, 3.0 s pad/trunc -> log-mel (torchaudio,
         64 mels, n_fft 1024, hop 256).
Video  : .flv (OpenCV/ffmpeg) -> 8 uniformly-sampled frames, center-cropped
         64x64 grayscale in [0,1].
No new heavy dependencies (torchaudio.transforms + scipy.io.wavfile + cv2).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import wavfile

import torchaudio

SR = 16000
DUR = 3.0
N_MELS = 64
N_FFT = 1024
HOP = 256
N_FRAMES = 8
FRAME_SIZE = 64
T_FIXED = int(DUR * SR) // HOP + 1        # 188

_MEL = None


def _mel():
    global _MEL
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return _MEL


def load_audio_logmel(path: str) -> torch.Tensor:
    """Return log-mel [1, 64, T_FIXED]. Raises on undecodable audio."""
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    # normalize integer PCM to [-1, 1]
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    if peak > 1.0:
        data = data / 32768.0
    wav = torch.from_numpy(data).float()[None, :]                 # [1, n]
    if rate != SR:
        wav = torchaudio.functional.resample(wav, rate, SR)
    n = int(DUR * SR)
    if wav.shape[1] < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[1]))
    else:
        wav = wav[:, :n]
    mel = _mel()(wav)                                            # [1, 64, T]
    logmel = torch.log(mel + 1e-6)
    if logmel.shape[2] < T_FIXED:
        logmel = torch.nn.functional.pad(logmel, (0, T_FIXED - logmel.shape[2]))
    else:
        logmel = logmel[:, :, :T_FIXED]
    return logmel


def load_video_frames(path: str) -> torch.Tensor:
    """Return [N_FRAMES, 1, 64, 64] grayscale in [0,1]. Raises if no frames."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"no decodable frames: {path}")
    idxs = np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)
    out = []
    for i in idxs:
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        s = min(h, w)
        g = g[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]   # center square
        g = cv2.resize(g, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        out.append(g.astype(np.float32) / 255.0)
    arr = np.stack(out)[:, None, :, :]                          # [N,1,64,64]
    return torch.from_numpy(arr).float()


class CremadDataset(torch.utils.data.Dataset):
    """Reads the frozen manifest; returns (logmel, frames, label). On-the-fly
    decode (fine for smoke / small runs); a cache can be added for full R1-R3."""

    def __init__(self, manifest, split: str, limit: int | None = None):
        import pandas as pd
        df = manifest if hasattr(manifest, "iloc") else __import__("pandas").read_csv(manifest)
        df = df[df.split == split].reset_index(drop=True)
        if limit:
            # keep class balance in the tiny subset
            df = df.groupby("label", group_keys=False).apply(
                lambda g: g.head(max(1, limit // df.label.nunique()))).reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        logmel = load_audio_logmel(r.audio_path)
        frames = load_video_frames(r.video_path)
        return logmel, frames, int(r.label), r.actor_id
