"""Generate EXTERNAL test inputs for the Prediction Studio (nothing here comes from CREMA-D
or UTD-MHAD content, except the clearly-labelled UTD format-conversion files).

  speech_tts.wav        Windows SAPI text-to-speech (22.05 kHz) -- real speech, not in CREMA-D
  speech_tts.{flac,ogg,m4a,webm,mp3}   the same speech in other containers/codecs
  speech_tts_long.wav   ~10 s of TTS speech (tests the 3.0 s crop)
  speech_tts_short.wav  0.3 s cut (tests "too short")
  face_still.mp4        a real human face photo (skimage astronaut) as a still 2 s video
  face_zoom.webm        the same photo with a slow zoom (still not a speaking face)
  face_zoom.flv         the same zoom as FLV1 in .flv (browsers cannot play FLV; tests the server preview)
  face_with_speech.mp4  face_zoom + speech_tts muxed as the video's own audio track
  no_face.mp4           a photo with no face (skimage coffee)
  dark_face.mp4         face_zoom darkened to ~3% brightness
  noise.wav, tone.wav, silence.wav     non-speech audio
  corrupted.wav, corrupted.mp4, not_audio.txt   invalid files
  utd_conv/  a UTD-MHAD TEST-split trial re-saved as .npy depth and .csv inertial
             (format conversion of in-dataset data -- NOT external content),
             plus the same inertial signal converted to m/s^2 (units-mismatch example)

Usage: python scripts/make_external_samples.py [out_dir]   (default samples/external)
"""
from __future__ import annotations

import glob
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av
import cv2
import numpy as np
import scipy.io as sio
from scipy.io import wavfile

SENTENCE = "I really cannot believe you would do something like that to me again."
LONG = ("I really cannot believe you would do something like that to me again. "
        "We talked about this last week, and you promised me it would never happen.")


def tts(text: str, out: Path, voice: str = "Microsoft Zira Desktop") -> None:
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); $s.SetOutputToWaveFile('{out}'); $s.Speak(\"{text}\"); $s.Dispose()")
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def encode_audio(wav: Path, out: Path, container: str, codec: str, rate: int | None = None) -> None:
    sr, x = wavfile.read(wav)
    x = x.astype(np.int16)
    with av.open(str(out), "w", format=container) as c:
        st = c.add_stream(codec, rate=rate or sr)
        st.layout = "mono"
        fr = av.AudioFrame.from_ndarray(x[None, :], format="s16", layout="mono")
        fr.sample_rate = sr
        rs = av.AudioResampler(format=st.format.name, layout="mono", rate=st.rate)
        for f in rs.resample(fr) + rs.resample(None):
            for p in st.encode(f):
                c.mux(p)
        for p in st.encode(None):
            c.mux(p)


def write_video(frames_rgb: list, out: Path, fps: int, codec: str, container: str, audio_wav: Path | None = None) -> None:
    h, w = frames_rgb[0].shape[:2]
    with av.open(str(out), "w", format=container) as c:
        vs = c.add_stream(codec, rate=fps)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        ast = None
        if audio_wav is not None:
            sr, x = wavfile.read(audio_wav)
            ast = c.add_stream("aac", rate=sr)
            ast.layout = "mono"
        for f in frames_rgb:
            for p in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in vs.encode(None):
            c.mux(p)
        if ast is not None:
            fr = av.AudioFrame.from_ndarray(x.astype(np.int16)[None, :], format="s16", layout="mono")
            fr.sample_rate = sr
            rs = av.AudioResampler(format=ast.format.name, layout="mono", rate=sr)
            for f in rs.resample(fr) + rs.resample(None):
                for p in ast.encode(f):
                    c.mux(p)
            for p in ast.encode(None):
                c.mux(p)


def zoom_frames(img: np.ndarray, n: int, size: tuple[int, int], z0=1.0, z1=1.25) -> list:
    h, w = img.shape[:2]
    out = []
    for k in range(n):
        z = z0 + (z1 - z0) * k / max(1, n - 1)
        ch, cw = int(h / z), int(w / z)
        y, x = (h - ch) // 2, (w - cw) // 2
        out.append(cv2.resize(img[y:y + ch, x:x + cw], size, interpolation=cv2.INTER_AREA))
    return out


def main(out: Path) -> None:
    from skimage import data
    out.mkdir(parents=True, exist_ok=True)
    tts(SENTENCE, out / "speech_tts.wav")
    tts(LONG, out / "speech_tts_long.wav", voice="Microsoft David Desktop")
    sr, x = wavfile.read(out / "speech_tts.wav")
    nz = np.flatnonzero(np.abs(x) > 500)
    wavfile.write(out / "speech_tts_short.wav", sr, x[nz[0]: nz[0] + int(0.3 * sr)])
    for ext, (cont, codec, rate) in {"flac": ("flac", "flac", None), "ogg": ("ogg", "libopus", 48000),
                                     "m4a": ("mp4", "aac", None), "webm": ("webm", "libopus", 48000),
                                     "mp3": ("mp3", "libmp3lame", None)}.items():
        encode_audio(out / "speech_tts.wav", out / f"speech_tts.{ext}", cont, codec, rate)
    t = np.arange(int(3.0 * 16000)) / 16000
    wavfile.write(out / "tone.wav", 16000, (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))
    rng = np.random.default_rng(0)
    wavfile.write(out / "noise.wav", 16000, (0.3 * rng.standard_normal(len(t)) * 32767).clip(-32767, 32767).astype(np.int16))
    wavfile.write(out / "silence.wav", 16000, np.zeros(len(t), dtype=np.int16))

    face = data.astronaut()                      # real photograph of a human face (RGB 512x512)
    face = face[:400, 60:460]                    # head-and-shoulders crop
    still = [cv2.resize(face, (480, 360), interpolation=cv2.INTER_AREA)] * 50
    write_video(still, out / "face_still.mp4", 25, "libx264", "mp4")
    zoom = zoom_frames(face, 60, (640, 480))
    write_video(zoom, out / "face_zoom.webm", 30, "libvpx", "webm")
    write_video(zoom, out / "face_zoom.flv", 30, "flv", "flv")
    write_video(zoom[:90] if len(zoom) >= 90 else zoom + zoom[::-1][:30], out / "face_with_speech.mp4", 30,
                "libx264", "mp4", audio_wav=out / "speech_tts.wav")
    write_video([(f * 0.03).astype(np.uint8) for f in zoom], out / "dark_face.mp4", 30, "libx264", "mp4")
    write_video(zoom_frames(data.coffee(), 60, (640, 480)), out / "no_face.mp4", 30, "libx264", "mp4")

    (out / "corrupted.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + rng.bytes(2000))
    (out / "corrupted.mp4").write_bytes((out / "face_still.mp4").read_bytes()[:1500])
    (out / "not_audio.txt").write_text("this is plainly not an audio file\n")

    conv = out / "utd_conv"; conv.mkdir(exist_ok=True)
    d = sorted(glob.glob(str(ROOT.parent / "UTD-MHAD" / "Depth" / "**" / "a3_s4_t2_depth.mat"), recursive=True))[0]
    i = sorted(glob.glob(str(ROOT.parent / "UTD-MHAD" / "Inertial" / "**" / "a3_s4_t2_inertial.mat"), recursive=True))[0]
    dep = sio.loadmat(d)["d_depth"]
    np.save(conv / "a3_s4_t2_depth.npy", np.moveaxis(dep, 2, 0))           # (T,H,W)
    ine = sio.loadmat(i)["d_iner"]
    np.savetxt(conv / "a3_s4_t2_inertial.csv", ine, delimiter=",",
               header="acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z", comments="")
    ms2 = ine.copy(); ms2[:, :3] *= 9.80665
    np.savetxt(conv / "a3_s4_t2_inertial_ms2.csv", ms2, delimiter=",",
               header="acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z", comments="")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples" / "external")
