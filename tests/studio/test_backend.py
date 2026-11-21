"""FastAPI backend: endpoints, error codes, temp-file cleanup, no persistence."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    with TestClient(app) as c:
        yield c


def _studio_tmp_dirs():
    return {p.name for p in Path(tempfile.gettempdir()).glob("studio_*")}


def test_health_and_models(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["models"]["cremad"]["loaded"] and h["models"]["utd_mhad"]["loaded"]
    m = client.get("/models").json()
    assert m["research_prototype"] is True
    assert m["models"]["cremad"]["labels"]["n_classes"] == 6
    assert "POST /predict/utd_mhad/multimodal" in m["endpoints"]


def test_audio_prediction_external(client, external):
    before = _studio_tmp_dirs()
    with open(external / "speech_tts.webm", "rb") as f:
        r = client.post("/predict/cremad/audio", files={"audio": ("rec.webm", f, "audio/webm")})
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    body = r.json()
    assert body["mode"] == "audio_only" and len(body["primary"]["probabilities"]) == 6
    assert _studio_tmp_dirs() <= before                       # per-request temp dir removed


def test_multimodal_with_separate_files(client, external):
    with open(external / "face_zoom.webm", "rb") as v, open(external / "speech_tts.wav", "rb") as a:
        r = client.post("/predict/cremad/multimodal", files={"video": ("v.webm", v), "audio": ("a.wav", a)})
    assert r.status_code == 200 and r.json()["mode"] == "multimodal"


def test_utd_endpoint(client, external):
    with open(external / "utd_conv/a3_s4_t2_depth.npy", "rb") as d, open(external / "utd_conv/a3_s4_t2_inertial.csv", "rb") as i:
        r = client.post("/predict/utd_mhad/multimodal", files={"depth": ("d.npy", d), "inertial": ("i.csv", i)})
    assert r.status_code == 200 and len(r.json()["primary"]["probabilities"]) == 27


@pytest.mark.parametrize("endpoint,files,status,code", [
    ("/predict/cremad/audio", {"audio": ("x.txt", b"definitely not audio, just text\n")}, 415, "unrecognized_format"),
    ("/predict/cremad/audio", {"audio": ("x.wav", b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunkjunk")}, 422, "audio_decode_failed"),
    ("/predict/cremad/video", {"video": ("x.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)}, 422, None),
    ("/predict/cremad/audio", {}, 400, "invalid_request"),
])
def test_error_responses(client, endpoint, files, status, code):
    r = client.post(endpoint, files=files)
    assert r.status_code == status
    err = r.json()["error"]
    if code:
        assert err["code"] == code
    assert err["message"]


def test_multimodal_without_audio_when_disallowed(client, external):
    with open(external / "face_zoom.webm", "rb") as v:
        r = client.post("/predict/cremad/multimodal", files={"video": ("v.webm", v)}, data={"use_video_audio": "false"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_modality"


def test_oversized_upload_rejected(client, monkeypatch):
    import inference.input_validator as iv
    monkeypatch.setitem(iv.LIMITS, "audio", 1000)
    r = client.post("/predict/cremad/audio", files={"audio": ("big.wav", b"RIFF" + b"\x00" * 5000)})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_no_checkpoint_upload_route(client):
    paths = {r.path for r in client.app.routes}
    assert not any("checkpoint" in p or "upload_model" in p for p in paths)
