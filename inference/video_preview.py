"""Server-side video preview for files the visitor's browser cannot play (e.g. CREMA-D's .flv).

Decodes with the SAME loader the predictor uses (`load_video_frames_streaming`), so the
frames returned are exactly the 8 frames the model would sample, with the same limits and
error codes. Nothing is stored; thumbnails go back in the response only.
"""
from __future__ import annotations

import base64
from pathlib import Path

import av
import cv2

from inference.cremad_preprocessor import load_video_frames_streaming

THUMB_W = 192          # display width of each returned thumbnail (px)
JPEG_QUALITY = 80


def _jpeg_b64(bgr) -> str:
    h, w = bgr.shape[:2]
    if w > THUMB_W:
        bgr = cv2.resize(bgr, (THUMB_W, max(1, round(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _audio_stream(path: Path) -> dict | None:
    """Describe the container's first audio stream (None if it has none or cannot be probed)."""
    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                return None
            a = c.streams.audio[0]
            dur = float(a.duration * a.time_base) if a.duration and a.time_base else None
            return {"codec": a.codec_context.name, "sample_rate": int(a.codec_context.sample_rate or 0),
                    "duration_s": dur}
    except Exception:  # noqa: BLE001 -- a probe failure only means "unknown"
        return None


def video_preview(path: Path, request_id: str) -> dict:
    v = load_video_frames_streaming(path, role="video")
    s = min(v.width, v.height)
    crop = {"x": (v.width - s) / 2 / v.width, "y": (v.height - s) / 2 / v.height,
            "w": s / v.width, "h": s / v.height} if v.width and v.height else None
    frames = [{"index": int(i), "time_s": (i / v.fps if v.fps > 0 else None), "image": _jpeg_b64(f)}
              for i, f in zip(v.indices, v.selected_bgr)]
    return {"request_id": request_id, "n_frames": v.n_frames, "fps": v.fps, "width": v.width,
            "height": v.height, "duration_s": v.duration_s, "crop": crop, "frames": frames,
            "audio_stream": _audio_stream(path)}
