"""Input handling, quality/domain checks, status logic and external samples (engine level)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference.input_validator import InputError, check_file


def _audio(pred, f: Path, tmp):
    from inference.input_validator import check_file
    return pred.predict(request_id="t", workdir=tmp, audio=(f, check_file(f, "audio").kind))


def _video(pred, f: Path, tmp):
    return pred.predict(request_id="t", workdir=tmp, video=(f, check_file(f, "video").kind))


# ---------------------------------------------------------------- file validation
@pytest.mark.parametrize("name,role,code", [("not_audio.txt", "audio", "unrecognized_format"),
                                            ("speech_tts.wav", "video", "wrong_content_type"),
                                            ("face_zoom.webm", "depth", "wrong_content_type")])
def test_unsupported_formats_rejected(external, name, role, code):
    with pytest.raises(InputError) as e:
        check_file(external / name, role)
    assert e.value.code == code and e.value.http_status == 415


@pytest.mark.parametrize("name,role", [("1003_IWL_SAD_XX.wav", "audio"), ("1001_DFA_SAD_XX.flv", "video"),
                                       ("a01_s1_t1_inertial.mat", "inertial"), ("a01_s1_t1_depth.mat", "depth")])
def test_git_lfs_pointer_rejected_with_explanation(tmp_path, name, role):
    # Exact layout of a Git LFS pointer as found in a clone made without `git lfs pull`.
    f = tmp_path / name
    f.write_bytes(b"version https://git-lfs.github.com/spec/v1\n"
                  b"oid sha256:8ccd279007363fdcc3061f4491e4e64c05120703f0c3ba72c31a8fa503ddb575\n"
                  b"size 119630\n")
    with pytest.raises(InputError) as e:
        check_file(f, role)
    assert e.value.code == "lfs_pointer" and e.value.http_status == 415 and e.value.field == role
    assert "Git LFS placeholder" in e.value.message and "119,630 bytes" in e.value.message
    assert "delimited text" not in e.value.message


def test_empty_and_oversized(tmp_path, monkeypatch):
    import inference.input_validator as iv
    f = tmp_path / "e.wav"; f.write_bytes(b"")
    with pytest.raises(InputError, match="empty"):
        check_file(f, "audio")
    f.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100)
    monkeypatch.setitem(iv.LIMITS, "audio", 50)
    with pytest.raises(InputError) as e:
        check_file(f, "audio")
    assert e.value.http_status == 413


@pytest.mark.parametrize("name,code", [("corrupted.wav", "audio_decode_failed")])
def test_corrupted_audio(cremad_predictor, external, tmp_path, name, code):
    with pytest.raises(InputError) as e:
        _audio(cremad_predictor, external / name, tmp_path)
    assert e.value.code == code


def test_corrupted_video(cremad_predictor, external, tmp_path):
    with pytest.raises(InputError) as e:
        _video(cremad_predictor, external / "corrupted.mp4", tmp_path)
    assert e.value.code in {"video_decode_failed", "no_decodable_frames"}


def test_missing_audio_stream_for_multimodal(cremad_predictor, external, tmp_path):
    with pytest.raises(InputError) as e:
        cremad_predictor.predict(request_id="t", workdir=tmp_path, video=(external / "face_zoom.webm", "webm"),
                                 audio_from_video=True)
    assert e.value.code == "missing_audio_stream"


def test_no_input_at_all(cremad_predictor, tmp_path):
    with pytest.raises(InputError):
        cremad_predictor.predict(request_id="t", workdir=tmp_path)


def test_utd_requires_both_modalities(utd_predictor, external, tmp_path):
    with pytest.raises(InputError) as e:
        utd_predictor.predict(request_id="t", workdir=tmp_path, depth=(external / "utd_conv/a3_s4_t2_depth.npy", "npy"),
                              inertial=None)
    assert e.value.code == "missing_modality"


# ---------------------------------------------------------------- domain / quality outcomes
@pytest.mark.parametrize("name,code", [("silence.wav", "silent_audio"), ("tone.wav", "no_speech_detected"),
                                       ("noise.wav", "no_speech_detected"), ("speech_tts_short.wav", "audio_too_short")])
def test_non_speech_and_degenerate_audio_is_not_classified(cremad_predictor, external, tmp_path, name, code):
    r = _audio(cremad_predictor, external / name, tmp_path)
    assert r.status == "unsupported" and r.primary is None and r.unimodal == {}
    assert code in {i.code for i in r.issues}


def test_long_clip_is_truncated_like_training(cremad_predictor, external, tmp_path):
    r = _audio(cremad_predictor, external / "speech_tts_long.wav", tmp_path)
    assert r.primary is not None and "audio_truncated" in {i.code for i in r.issues}
    assert r.input_report["audio"]["duration_s"] > 5


@pytest.mark.parametrize("name,code,status", [("no_face.mp4", "no_face_detected", "unsupported"),
                                              ("dark_face.mp4", "video_too_dark", "unsupported"),
                                              ("face_still.mp4", "video_static", "low_quality")])
def test_video_quality_and_domain(cremad_predictor, external, tmp_path, name, code, status):
    r = _video(cremad_predictor, external / name, tmp_path)
    assert r.status == status and code in {i.code for i in r.issues}
    assert (r.primary is None) == (status == "unsupported")


def test_utd_units_warning(utd_predictor, external, tmp_path):
    r = utd_predictor.predict(request_id="t", workdir=tmp_path,
                              depth=(external / "utd_conv/a3_s4_t2_depth.npy", "npy"),
                              inertial=(external / "utd_conv/a3_s4_t2_inertial_ms2.csv", "csv"))
    assert "inertial_units_accel" in {i.code for i in r.issues} and r.status == "low_quality"


# ---------------------------------------------------------------- external samples end-to-end
@pytest.mark.parametrize("name", ["speech_tts.wav", "speech_tts.flac", "speech_tts.ogg", "speech_tts.m4a",
                                  "speech_tts.webm", "speech_tts.mp3"])
def test_external_speech_in_every_container_is_classified(cremad_predictor, external, tmp_path, name):
    r = _audio(cremad_predictor, external / name, tmp_path)
    assert r.mode == "audio_only" and r.primary is not None
    assert r.primary.source == "frozen_encoder_probe" and r.unimodal.keys() == {"audio"}
    p = np.array([c.probability for c in r.primary.probabilities])
    assert p.shape == (6,) and abs(p.sum() - 1) < 1e-9


def test_external_face_with_own_speech_track_multimodal(cremad_predictor, external, tmp_path):
    r = cremad_predictor.predict(request_id="t", workdir=tmp_path, video=(external / "face_with_speech.mp4", "mp4"),
                                 audio_from_video=True)
    assert r.mode == "multimodal" and r.primary.source == "joint_head"
    assert r.unimodal.keys() == {"audio", "video"}
    assert r.input_report["audio"]["source"] == "audio track of the uploaded video"
    assert any(s.detector == "mahalanobis:joint" and s.used_for_status for s in r.shift)
    names = [l["name"] for l in r.trace["layers"]]
    assert names[0] == "enc_a.net.0.0" and names[-1] == "head" and "fusion (sum)" in names


def test_unimodal_never_runs_the_missing_branch(cremad_predictor, external, tmp_path):
    r = _audio(cremad_predictor, external / "speech_tts.wav", tmp_path)
    assert all(l["name"].startswith("enc_a") for l in r.trace["layers"])
    r = _video(cremad_predictor, external / "face_zoom.webm", tmp_path)
    assert all(l["name"].startswith("enc_v") for l in r.trace["layers"])


# ---------------------------------------------------------------- status logic
def test_status_priority():
    from inference.common import decide_status
    from inference.prediction_schema import Issue, ShiftAssessment
    sh = ShiftAssessment(detector="d", score=10, threshold=1, exceeds=True, ratio_to_threshold=10,
                         validation_flag_rate=.05, test_flag_rate_evaluation_only=.05, used_for_status=True)
    diag = sh.model_copy(update={"used_for_status": False})
    lq = Issue(code="x", severity="low_quality", modality="audio", message="m")
    un = Issue(code="y", severity="unsupported", modality="audio", message="m")
    assert decide_status([], [])[0] == "compatible"
    assert decide_status([], [diag])[0] == "compatible"          # diagnostic detectors never set status
    assert decide_status([], [sh])[0] == "potential_distribution_shift"
    assert decide_status([lq], [sh])[0] == "low_quality"
    assert decide_status([lq, un], [sh])[0] == "unsupported"


def test_detector_flags_far_representations(cremad_pkg):
    det = cremad_pkg.detectors["audio"]
    mu = det.means[0]
    assert not det.assess(mu[None])["exceeds"]
    assert det.assess((mu + 50.0)[None])["exceeds"]
