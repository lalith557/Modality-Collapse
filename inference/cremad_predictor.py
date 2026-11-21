"""CREMA-D predictor: audio-only, video-only and audio+video inference.

* multimodal  -> primary = the trained joint head on (audio, video); the two frozen-encoder
                 probes are also reported for modality comparison.
* audio_only  -> the audio frozen-encoder probe; the video branch is NOT run and nothing is
                 substituted for the missing modality.
* video_only  -> the video frozen-encoder probe, likewise.
Inputs flagged "unsupported" by the domain checks are not classified at all.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from inference.common import (Clock, decide_status, limitations, make_prediction, minmax, model_info, png_b64,
                              probe_prediction, reference_accuracy, shift_assessment, traced)
from inference.cremad_preprocessor import (audio_tensor, extract_audio_track_from_video, load_video_frames_streaming,
                                           prepare_audio)
from inference.input_validator import InputError
from inference.model_loader import Package
from inference.prediction_schema import Issue, PredictionResponse, TraceLayer
from inference.quality_checks import assess_audio, assess_video, audio_metrics, video_metrics


class CremadPredictor:
    def __init__(self, pkg: Package):
        if pkg.dataset != "cremad":
            raise ValueError("CremadPredictor needs the cremad package")
        self.pkg = pkg
        self.lock = threading.Lock()          # serialize GPU forward passes (hooks are per-call)

    def _rate(self, modality: str, code: str):
        q = self.pkg.quality_reference or {}
        return q.get("false_flag_rates", {}).get("test_evaluation_only", {}).get(modality, {}) \
                .get("per_check_rate", {}).get(code)

    def predict(self, *, request_id: str, workdir: Path, audio: tuple[Path, str] | None = None,
                video: tuple[Path, str] | None = None, audio_from_video: bool = False) -> PredictionResponse:
        pkg, dev = self.pkg, self.pkg.device
        if pkg.quality_reference is None:
            raise RuntimeError("quality_reference.json missing; run scripts/calibrate_studio_quality.py")
        clock = Clock()
        issues: list[Issue] = []
        report, pre = {}, {}
        xa = xv = None
        if video is None and audio is None and not audio_from_video:
            raise InputError("no_input", "Provide an audio file, a video file, or both.", 400)

        if video is not None:
            with clock.step("decode_video", "OpenCV/FFmpeg; 8 frames at np.linspace(0, n-1, 8).astype(int)"):
                vi = load_video_frames_streaming(video[0])
            with clock.step("video_quality_checks", "face (Haar, centre crop), blur, brightness, motion"):
                vm = video_metrics(vi.selected_bgr, vi.tensor, vi.width, vi.height, vi.n_frames, vi.fps, vi.duration_s)
                for i in assess_video(vm, pkg.quality_reference):
                    issues.append(Issue(modality="video", measured_false_flag_rate_test=self._rate("video", i["code"]), **i))
            report["video"] = {**vm, "container": video[1], "frame_indices": vi.indices}
            pre["video_frames"] = [png_b64(f[0]) for f in vi.tensor.numpy()]
            pre["video_tensor_shape"] = [1, *vi.tensor.shape]
            xv = vi.tensor[None].to(dev)

        if audio is not None or audio_from_video:
            with clock.step("decode_audio", "WAV direct, or PyAV -> 16-bit PCM WAV at native rate"):
                if audio is not None:
                    ai = prepare_audio(audio[0], audio[1], workdir)
                    src = "separate audio file"
                else:
                    ai = extract_audio_track_from_video(video[0], workdir)
                    src = "audio track of the uploaded video"
            with clock.step("audio_quality_checks", "silence, speech activity, clipping, noise, duration"):
                am = audio_metrics(ai.mono_native, ai.native_rate)
                for i in assess_audio(am, pkg.quality_reference):
                    issues.append(Issue(modality="audio", measured_false_flag_rate_test=self._rate("audio", i["code"]), **i))
            with clock.step("audio_preprocessing", "data.crema_d.load_audio_logmel (training function) + float16"):
                logmel = audio_tensor(ai)                                    # [1,64,188]
            report["audio"] = {**am, "source": src, "decoder": ai.source, "codec": ai.codec,
                               "native_sample_rate": ai.native_rate, "channels": ai.channels}
            pre["audio_logmel"] = png_b64(minmax(np.flipud(logmel[0].numpy())), scale=2)
            pre["audio_tensor_shape"] = [1, *logmel.shape]
            pre["audio_logmel_range"] = [float(logmel.min()), float(logmel.max())]
            xa = logmel[None].to(dev)

        mode = "multimodal" if (xa is not None and xv is not None) else ("audio_only" if xa is not None else "video_only")
        primary, unimodal, shift, layers, reps = None, {}, [], [], {}
        if not any(i.severity == "unsupported" for i in issues):
            m = pkg.model
            with self.lock, torch.inference_mode(), clock.step("model_forward", f"{mode}; eval mode, inference_mode"):
                with traced(m, layers):
                    if mode == "multimodal":
                        logits, (h_a, h_v) = m(xa, xv, return_reps=True)
                    elif mode == "audio_only":
                        h_a = m.norm_a(m.enc_a(xa))
                    else:
                        h_v = m.norm_v(m.enc_v(xv))
                if mode == "multimodal":
                    z = h_a + h_v
                    fusion = TraceLayer(name="fusion (sum)", type="models.fusion.fuse('sum')", output_shape=list(z.shape),
                                        mean=float(z.mean()), std=float(z.std()), min=float(z.min()), max=float(z.max()))
                    layers.insert(len(layers) - 1, fusion)
                    primary = make_prediction(logits, pkg, "joint_head", ["audio", "video"],
                                              "Trained joint model: both encoders, sum fusion, linear head.",
                                              reference_accuracy(pkg, "joint"))
                    unimodal = {"audio": probe_prediction(pkg, "audio", h_a), "video": probe_prediction(pkg, "video", h_v)}
                    joint = np.concatenate([h_a.cpu().numpy(), h_v.cpu().numpy()], 1)
                    shift = [shift_assessment(pkg, "joint", joint, True),
                             shift_assessment(pkg, "audio", h_a.cpu().numpy(), False),
                             shift_assessment(pkg, "video", h_v.cpu().numpy(), False)]
                    reps = {"h_audio": h_a[0].tolist(), "h_video": h_v[0].tolist(), "fused_z": z[0].tolist(),
                            "logits": logits[0].tolist()}
                elif mode == "audio_only":
                    primary = probe_prediction(pkg, "audio", h_a); unimodal = {"audio": primary}
                    shift = [shift_assessment(pkg, "audio", h_a.cpu().numpy(), True)]
                    reps = {"h_audio": h_a[0].tolist()}
                else:
                    primary = probe_prediction(pkg, "video", h_v); unimodal = {"video": primary}
                    shift = [shift_assessment(pkg, "video", h_v.cpu().numpy(), True)]
                    reps = {"h_video": h_v[0].tolist()}

        status, reasons = decide_status(issues, shift)
        trace = {"executed": primary is not None, "layers": [l.model_dump() for l in layers],
                 "representations": reps,
                 "notes": ["Shapes and statistics are recorded by forward hooks on the real model during this request.",
                           "VideoCNN averages its per-frame features over the 8 frames inside VideoCNN.forward, "
                           "between enc_v.net.4 and enc_v.fc; that step has no module, so no hook.",
                           "norm_a / norm_v are Identity and the scale injector is disabled in the baseline."]}
        return PredictionResponse(request_id=request_id, dataset="cremad", mode=mode, status=status,
                                  status_reasons=reasons, primary=primary, unimodal=unimodal, issues=issues,
                                  shift=shift, input_report=report, preprocessed=pre, trace=trace,
                                  steps=clock.steps, total_ms=clock.total_ms, model=model_info(pkg),
                                  limitations=limitations(pkg))
