"""UTD-MHAD predictor: depth + inertial (both required).

Primary = trained joint head; depth-only and inertial-only frozen-encoder probes are also
reported. The shift status uses the joint detector; the depth detector is diagnostic only
because it flags ~1/3 of in-distribution test clips (see evaluation.json).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from inference.common import (Clock, decide_status, limitations, make_prediction, minmax, model_info, png_b64,
                              probe_prediction, reference_accuracy, shift_assessment, traced)
from inference.input_validator import InputError
from inference.model_loader import Package
from inference.prediction_schema import Issue, PredictionResponse, TraceLayer
from inference.quality_checks import assess_depth, assess_inertial
from inference.utd_preprocessor import prepare_depth, prepare_inertial


class UtdPredictor:
    def __init__(self, pkg: Package):
        if pkg.dataset != "utd_mhad":
            raise ValueError("UtdPredictor needs the utd_mhad package")
        self.pkg = pkg
        self.lock = threading.Lock()
        self.mean = np.asarray(pkg.norm_stats["inertial_channel_mean"], dtype=np.float64)
        self.std = np.asarray(pkg.norm_stats["inertial_channel_std"], dtype=np.float64)

    def predict(self, *, request_id: str, workdir: Path, depth: tuple[Path, str] | None,
                inertial: tuple[Path, str] | None) -> PredictionResponse:
        if depth is None or inertial is None:
            raise InputError("missing_modality", "UTD-MHAD inference needs BOTH a depth file and an inertial file.",
                             400, "depth" if depth is None else "inertial")
        pkg, dev = self.pkg, self.pkg.device
        clock = Clock()
        issues: list[Issue] = []
        with clock.step("decode_depth", "data.utd_mhad.load_depth_clip (training function)"):
            di = prepare_depth(depth[0], depth[1], workdir)
        with clock.step("decode_inertial", "data.utd_mhad.load_inertial + TRAIN z-score + float16"):
            ii = prepare_inertial(inertial[0], inertial[1], workdir, self.mean, self.std)
        with clock.step("quality_checks", "shape, units and value sanity"):
            for i in assess_depth(di.raw_thw):
                issues.append(Issue(modality="depth", **i))
            for i in assess_inertial(ii.raw_t6):
                issues.append(Issue(modality="inertial", **i))
        report = {"depth": {"raw_shape": list(di.raw_shape), "frames": int(di.raw_thw.shape[0]),
                            "value_range": [float(di.raw_thw.min()), float(di.raw_thw.max())], "format": di.source_kind},
                  "inertial": {"samples": ii.n_samples, "format": ii.source_kind,
                               "channel_means_raw": ii.raw_t6.mean(0).tolist(),
                               "channel_std_raw": ii.raw_t6.std(0).tolist(),
                               "zscore_abs_max": float(np.abs(ii.tensor.numpy()).max())}}
        pre = {"depth_frames": [png_b64(f[0], scale=2) for f in di.tensor.numpy()],
               "depth_tensor_shape": [1, *di.tensor.shape],
               "inertial_zscored": ii.tensor.numpy().round(4).tolist(),
               "inertial_tensor_shape": [1, *ii.tensor.shape],
               "inertial_channels": ["acc_x (g)", "acc_y (g)", "acc_z (g)", "gyro_x (deg/s)", "gyro_y (deg/s)", "gyro_z (deg/s)"]}
        xa, xv = di.tensor[None].to(dev), ii.tensor[None].to(dev)
        primary, unimodal, shift, layers, reps = None, {}, [], [], {}
        if not any(i.severity == "unsupported" for i in issues):
            m = pkg.model
            with self.lock, torch.inference_mode(), clock.step("model_forward", "multimodal; eval mode, inference_mode"):
                with traced(m, layers):
                    logits, (h_a, h_v) = m(xa, xv, return_reps=True)
                z = h_a + h_v
                layers.insert(len(layers) - 1, TraceLayer(
                    name="fusion (sum)", type="models.fusion.fuse('sum')", output_shape=list(z.shape),
                    mean=float(z.mean()), std=float(z.std()), min=float(z.min()), max=float(z.max())))
                primary = make_prediction(logits, pkg, "joint_head", ["depth", "inertial"],
                                          "Trained joint model: depth + inertial encoders, sum fusion, linear head.",
                                          reference_accuracy(pkg, "joint"))
                unimodal = {"depth": probe_prediction(pkg, "depth", h_a), "inertial": probe_prediction(pkg, "inertial", h_v)}
                joint = np.concatenate([h_a.cpu().numpy(), h_v.cpu().numpy()], 1)
                shift = [shift_assessment(pkg, "joint", joint, True),
                         shift_assessment(pkg, "depth", h_a.cpu().numpy(), False),
                         shift_assessment(pkg, "inertial", h_v.cpu().numpy(), False)]
                reps = {"h_depth": h_a[0].tolist(), "h_inertial": h_v[0].tolist(), "fused_z": z[0].tolist(),
                        "logits": logits[0].tolist()}
        status, reasons = decide_status(issues, shift)
        trace = {"executed": primary is not None, "layers": [l.model_dump() for l in layers], "representations": reps,
                 "notes": ["Shapes and statistics are recorded by forward hooks on the real model during this request.",
                           "DepthCNN averages per-frame features over the 16 frames inside DepthCNN.forward "
                           "(no module, so no hook) before enc_a.fc."]}
        return PredictionResponse(request_id=request_id, dataset="utd_mhad", mode="multimodal", status=status,
                                  status_reasons=reasons, primary=primary, unimodal=unimodal, issues=issues,
                                  shift=shift, input_report=report, preprocessed=pre, trace=trace,
                                  steps=clock.steps, total_ms=clock.total_ms, model=model_info(pkg),
                                  limitations=limitations(pkg))
