"""CREMA-D preprocessing for external inputs -- training-identical.

Audio: the model input is produced by the UNCHANGED training function
`data.crema_d.load_audio_logmel`. WAV uploads go to it directly. Any other container
(FLAC, MP3, Ogg, M4A, WebM, or the audio track of a video) is first decoded by PyAV to
16-bit PCM WAV at its *native* sample rate and channel layout -- no resampling, no gain
change -- and then passed to the same function, which performs the training-time mono
mix, int16 scaling, torchaudio resampling, 3.0 s crop/pad, log-mel and time crop.

Video: `load_video_frames_streaming` reproduces `data.crema_d.load_video_frames` exactly
(same frame count, `linspace(...).astype(int)` indices, BGR->GRAY, centre-square crop,
INTER_AREA resize to 64x64, /255) but decodes in two passes so that only the 8 selected
frames are held in memory. Bit-identity with the original is tested.

Both tensors get the float16 round-trip that the training cache applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from data.crema_d import (DUR, FRAME_SIZE, N_FRAMES, SR, load_audio_logmel)
from inference.input_validator import InputError

MAX_AUDIO_DECODE_S = 60.0      # decode at most this much of a non-WAV upload
MAX_VIDEO_S = 30.0             # longer videos are rejected (training clips are ~1.5-5 s)
MAX_VIDEO_FRAMES = 2400


@dataclass
class AudioInput:
    wav_path: Path                 # file actually fed to load_audio_logmel
    source: str                    # "wav-direct" | "pyav-decoded"
    codec: str
    native_rate: int
    channels: int
    duration_s: float
    mono_native: np.ndarray = field(repr=False)   # training-style mono float waveform, native rate


@dataclass
class VideoInput:
    tensor: torch.Tensor           # [8,1,64,64] float32 after float16 round-trip
    selected_bgr: list = field(repr=False)        # the 8 selected original-resolution frames
    n_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    indices: list = field(default_factory=list)


def _training_mono(data: np.ndarray) -> np.ndarray:
    """Exactly the first steps of load_audio_logmel: mono mean, float32, int16 scaling."""
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    if peak > 1.0:
        data = data / 32768.0
    return data


def _decode_with_pyav(src: Path, out_wav: Path, role: str) -> tuple[str, int, int]:
    import av
    try:
        container = av.open(str(src))
    except Exception as e:  # noqa: BLE001
        raise InputError("audio_decode_failed", f"Could not open the {role} file: {type(e).__name__}.", 422, role)
    with container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise InputError("missing_audio_stream", "The file contains no audio stream.", 422, role)
        st = streams[0]
        rate = int(st.codec_context.sample_rate or st.rate or 0)
        layout = st.codec_context.layout.name if st.codec_context.layout else "mono"
        if rate <= 0:
            raise InputError("audio_decode_failed", "Audio stream reports no sample rate.", 422, role)
        # Format conversion only (to packed s16); rate and layout are kept native.
        resampler = av.AudioResampler(format="s16", layout=layout, rate=rate)
        chunks, n = [], 0
        try:
            for frame in container.decode(st):
                for out in resampler.resample(frame):
                    arr = out.to_ndarray()                      # packed s16: [1, samples*channels]
                    chunks.append(arr.reshape(-1))
                    n += arr.size
                if n / max(1, st.codec_context.channels or 1) / rate > MAX_AUDIO_DECODE_S:
                    break
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
        except Exception as e:  # noqa: BLE001
            raise InputError("audio_decode_failed", f"Audio decoding failed ({type(e).__name__}).", 422, role)
        ch = max(1, int(st.codec_context.channels or 1))
        if not chunks:
            raise InputError("empty_audio", "The audio stream decoded to zero samples.", 422, role)
        pcm = np.concatenate(chunks).astype(np.int16)
        pcm = pcm[: (len(pcm) // ch) * ch].reshape(-1, ch)
        if pcm.shape[0] == 0:
            raise InputError("empty_audio", "The audio stream decoded to zero samples.", 422, role)
        wavfile.write(str(out_wav), rate, pcm if ch > 1 else pcm[:, 0])
        return st.codec_context.name, rate, ch


def prepare_audio(src: Path, kind: str, workdir: Path, role: str = "audio") -> AudioInput:
    """Make the file that load_audio_logmel will read, and the mono waveform for QC."""
    source, codec = "wav-direct", "pcm (WAV)"
    wav = src
    if kind == "wav":
        try:
            rate, data = wavfile.read(str(src))
        except Exception:  # noqa: BLE001 -- e.g. WAV codecs scipy cannot read
            kind = "other"
    if kind != "wav":
        wav = workdir / f"{role}_decoded.wav"
        codec, _, _ = _decode_with_pyav(src, wav, role)
        source = "pyav-decoded"
        rate, data = wavfile.read(str(wav))
    data = np.asarray(data)
    if data.size == 0:
        raise InputError("empty_audio", "The audio contains no samples.", 422, role)
    ch = 1 if data.ndim == 1 else int(data.shape[1])
    mono = _training_mono(data)
    return AudioInput(wav_path=wav, source=source, codec=codec, native_rate=int(rate), channels=ch,
                      duration_s=len(mono) / float(rate), mono_native=mono)


def audio_tensor(a: AudioInput) -> torch.Tensor:
    """[1,64,188] via the unchanged training function, then the cache's float16 round-trip."""
    try:
        x = load_audio_logmel(str(a.wav_path))
    except Exception as e:  # noqa: BLE001
        raise InputError("audio_decode_failed", f"Training preprocessing failed ({type(e).__name__}).", 422, "audio")
    return x.half().float()


def _process_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Per-frame body of load_video_frames, verbatim."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = min(h, w)
    g = g[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    g = cv2.resize(g, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0


def load_video_frames_streaming(path: str | Path, role: str = "video") -> VideoInput:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError("video_decode_failed", "The video could not be opened by the decoder.", 422, role)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = 0
    while cap.grab():                     # pass 1: count decodable frames (== read() loop count)
        n += 1
        if n > MAX_VIDEO_FRAMES:
            cap.release()
            raise InputError("video_too_long", f"The video has more than {MAX_VIDEO_FRAMES} frames; "
                             f"please upload a clip of at most {MAX_VIDEO_S:.0f} s.", 413, role)
    cap.release()
    if n == 0:
        raise InputError("no_decodable_frames", "No video frames could be decoded.", 422, role)
    duration = n / fps if fps > 0 else 0.0
    if duration > MAX_VIDEO_S:
        raise InputError("video_too_long", f"The video is {duration:.1f} s long; the limit is "
                         f"{MAX_VIDEO_S:.0f} s (training clips are about 1.5-5 s).", 413, role)
    idxs = np.linspace(0, n - 1, N_FRAMES).astype(int)          # identical to training
    wanted = {}
    for k, i in enumerate(idxs):
        wanted.setdefault(int(i), []).append(k)
    cap = cv2.VideoCapture(str(path))
    selected = [None] * N_FRAMES
    i = 0
    while i <= idxs[-1] and cap.grab():   # pass 2: retrieve only the selected frames
        if i in wanted:
            ok, f = cap.retrieve()
            if not ok:
                break
            for k in wanted[i]:
                selected[k] = f
        i += 1
    cap.release()
    if any(f is None for f in selected):
        raise InputError("video_decode_failed", "Selected frames could not be decoded.", 422, role)
    arr = np.stack([_process_frame(f) for f in selected])[:, None, :, :]
    t = torch.from_numpy(arr).float().half().float()
    if height == 0 or width == 0:
        height, width = selected[0].shape[:2]
    return VideoInput(tensor=t, selected_bgr=selected, n_frames=n, fps=fps, width=width,
                      height=height, duration_s=duration, indices=[int(x) for x in idxs])


def extract_audio_track_from_video(video_path: Path, workdir: Path) -> AudioInput:
    """Use a video's own audio stream (one recording -> both modalities)."""
    return prepare_audio(video_path, "container", workdir, role="audio")


__all__ = ["AudioInput", "VideoInput", "prepare_audio", "audio_tensor", "load_video_frames_streaming",
           "extract_audio_track_from_video", "SR", "DUR"]
