"""POST /preview/video: server-decoded frames for videos the browser cannot play (.flv/.avi).

The preview must show exactly the frames the model samples (same loader), store nothing,
log no content, and fail with the predictor's own error codes.
"""
from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    with TestClient(app) as c:
        yield c


def _studio_tmp_dirs():
    return {p.name for p in Path(tempfile.gettempdir()).glob("studio_*")}


def _decode(data_url: str) -> np.ndarray:
    head, b64 = data_url.split(",", 1)
    assert head == "data:image/jpeg;base64"
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def test_flv_preview_returns_the_frames_the_model_samples(client, external):
    from inference.cremad_preprocessor import load_video_frames_streaming
    from inference.video_preview import THUMB_W

    path = external / "face_zoom.flv"
    before = _studio_tmp_dirs()
    with open(path, "rb") as f:
        r = client.post("/preview/video", files={"video": ("1001_DFA_ANG_XX.flv", f, "video/x-flv")})
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    body = r.json()
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert _studio_tmp_dirs() <= before                       # per-request temp dir removed

    ref = load_video_frames_streaming(path, role="video")    # the predictor's own decode
    assert [f["index"] for f in body["frames"]] == [int(i) for i in ref.indices]
    assert len(body["frames"]) == 8
    assert (body["n_frames"], body["width"], body["height"]) == (ref.n_frames, ref.width, ref.height)
    assert body["fps"] == pytest.approx(ref.fps) and body["duration_s"] == pytest.approx(ref.duration_s)
    for f in body["frames"]:
        assert f["time_s"] == pytest.approx(f["index"] / ref.fps)
        img = _decode(f["image"])
        assert img.shape[1] == min(THUMB_W, ref.width)
        assert img.shape[0] == pytest.approx(ref.height * img.shape[1] / ref.width, abs=1)

    # centre square crop, as fractions of the frame
    s = min(ref.width, ref.height)
    assert body["crop"] == pytest.approx({"x": (ref.width - s) / 2 / ref.width, "y": (ref.height - s) / 2 / ref.height,
                                          "w": s / ref.width, "h": s / ref.height})
    # a preview is not a prediction
    assert not {"primary", "probabilities", "prediction"} & body.keys()


def test_frame_content_matches_the_decoded_frame(client, external):
    """Thumbnails are the decoded frames themselves (resized + JPEG), not placeholders."""
    from inference.cremad_preprocessor import load_video_frames_streaming

    path = external / "face_zoom.flv"
    with open(path, "rb") as f:
        body = client.post("/preview/video", files={"video": ("v.flv", f)}).json()
    ref = load_video_frames_streaming(path, role="video")
    for f, bgr in zip(body["frames"], ref.selected_bgr):
        thumb = _decode(f["image"]).astype(np.float32)
        want = cv2.resize(bgr, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        assert np.abs(thumb - want).mean() < 6.0            # JPEG q80 error only


def test_container_audio_stream_is_reported(client, external):
    with open(external / "face_with_speech.mp4", "rb") as f:
        body = client.post("/preview/video", files={"video": ("v.mp4", f, "video/mp4")}).json()
    a = body["audio_stream"]
    assert a is not None and a["codec"] and a["sample_rate"] > 0
    with open(external / "face_zoom.flv", "rb") as f:
        assert client.post("/preview/video", files={"video": ("v.flv", f)}).json()["audio_stream"] is None


@pytest.mark.parametrize("name,payload,status,code", [
    ("notes.txt", b"definitely not a video, just text\n", 415, "unrecognized_format"),
    ("broken.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, 422, None),
])
def test_preview_errors_use_the_predictor_codes(client, name, payload, status, code):
    before = _studio_tmp_dirs()
    r = client.post("/preview/video", files={"video": (name, payload)})
    assert r.status_code == status
    err = r.json()["error"]
    assert err["message"] and (code is None or err["code"] == code)
    assert r.headers["X-Request-ID"]
    assert _studio_tmp_dirs() <= before


def test_missing_file_is_a_400(client):
    r = client.post("/preview/video", files={})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_oversized_preview_rejected(client, monkeypatch):
    import inference.input_validator as iv
    monkeypatch.setitem(iv.LIMITS, "video", 1000)
    r = client.post("/preview/video", files={"video": ("big.flv", b"FLV\x01" + b"\x00" * 5000)})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_logs_carry_no_file_content(client, external, caplog):
    with caplog.at_level(logging.DEBUG, logger="studio"):
        with open(external / "face_zoom.flv", "rb") as f:
            body = client.post("/preview/video", files={"video": ("secret_name.flv", f)}).json()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "preview kind=" in text
    assert "base64" not in text and body["frames"][0]["image"][40:80] not in text
    assert "secret_name" not in text


def test_preview_is_listed_in_the_api(client):
    assert "POST /preview/video" in client.get("/models").json()["endpoints"]
