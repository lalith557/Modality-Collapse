"""Shared predictor machinery: traced forward passes, probability summaries, status logic."""
from __future__ import annotations

import base64
import math
import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch

from inference.model_loader import Package
from inference.prediction_schema import ClassProb, Issue, Prediction, ShiftAssessment, Step, TraceLayer
from inference.probes import probe_logits


class Clock:
    def __init__(self):
        self.steps: list[Step] = []
        self.t0 = time.perf_counter()

    @contextmanager
    def step(self, name: str, detail: str = ""):
        t = time.perf_counter()
        yield
        self.steps.append(Step(name=name, duration_ms=1000 * (time.perf_counter() - t), detail=detail))

    @property
    def total_ms(self) -> float:
        return 1000 * (time.perf_counter() - self.t0)


@contextmanager
def traced(model: torch.nn.Module, layers: list[TraceLayer]):
    """Record name/type/shape/statistics of every leaf module's REAL output in this forward pass."""
    handles = []
    for name, mod in model.named_modules():
        if not name or list(mod.children()) or isinstance(mod, torch.nn.Identity):
            continue

        def hook(m, _inp, out, name=name):
            o = out.detach().float()
            layers.append(TraceLayer(name=name, type=type(m).__name__, output_shape=list(o.shape),
                                     mean=float(o.mean()), std=float(o.std()) if o.numel() > 1 else 0.0,
                                     min=float(o.min()), max=float(o.max())))
        handles.append(mod.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def make_prediction(logits: torch.Tensor, pkg: Package, source: str, modalities: list[str],
                    description: str, reference: dict) -> Prediction:
    lg = logits.detach().double().reshape(-1)
    p = torch.softmax(lg, 0).cpu().numpy()
    k = int(p.argmax())
    cls = pkg.labels["classes"]
    ent = float(-(p * np.log(np.clip(p, 1e-300, 1))).sum())
    return Prediction(source=source, modalities_observed=modalities, description=description,
                      predicted_index=k, predicted_code=cls[k]["code"], predicted_name=cls[k]["name"],
                      probabilities=[ClassProb(index=c["index"], code=c["code"], name=c["name"],
                                               probability=float(p[c["index"]])) for c in cls],
                      max_probability=float(p[k]), entropy_nats=ent,
                      normalized_entropy=ent / math.log(len(cls)), reference_accuracy=reference)


def reference_accuracy(pkg: Package, which: str) -> dict:
    ev = pkg.evaluation
    block = ev["multimodal_joint_head"] if which == "joint" else ev["unimodal_frozen_encoder_probes"][which]
    return {"validation_accuracy": block["validation"]["accuracy"],
            "test_accuracy_evaluation_only": block["test_evaluation_only"]["accuracy"],
            "chance": 1.0 / pkg.labels["n_classes"]}


def probe_prediction(pkg: Package, modality: str, h: torch.Tensor) -> Prediction:
    return make_prediction(probe_logits(pkg.probes[modality], h), pkg, "frozen_encoder_probe", [modality],
                           f"{modality}-only: linear probe trained on the frozen {modality} encoder's "
                           "training-set representations (the paper's probe protocol). No other modality is "
                           "used or substituted.", reference_accuracy(pkg, modality))


def shift_assessment(pkg: Package, name: str, h: np.ndarray, used: bool) -> ShiftAssessment:
    det = pkg.detectors[name]
    a = det.assess(h)
    ev = pkg.evaluation["ood_detector"][name]
    return ShiftAssessment(detector=f"mahalanobis:{name}", score=a["score"], threshold=a["threshold"],
                           exceeds=a["exceeds"], ratio_to_threshold=a["ratio_to_threshold"],
                           validation_flag_rate=ev["validation_flag_rate"],
                           test_flag_rate_evaluation_only=ev["test_flag_rate_evaluation_only"], used_for_status=used)


def decide_status(issues: list[Issue], shift: list[ShiftAssessment]) -> tuple[str, list[str]]:
    reasons = [f"[{i.modality}] {i.message}" for i in issues if i.severity == "unsupported"]
    if reasons:
        return "unsupported", reasons
    lq = [f"[{i.modality}] {i.message}" for i in issues if i.severity == "low_quality"]
    sh = [f"Representation is far from the training distribution ({s.detector}: {s.ratio_to_threshold:.2f}x "
          f"the validation-calibrated threshold)." for s in shift if s.used_for_status and s.exceeds]
    if lq:
        return "low_quality", lq + sh
    if sh:
        return "potential_distribution_shift", sh
    return "compatible", ["All implemented checks passed. This does not mean the prediction is correct."]


def png_b64(arr01: np.ndarray, scale: int = 1) -> str:
    img = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    if scale > 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def model_info(pkg: Package) -> dict:
    m = pkg.metadata
    par = pkg.evaluation["parity_with_frozen_artifact"]
    return {"checkpoint_id": pkg.checkpoint_id, "checkpoint_sha256": m["checkpoint_hash"]["model.pth"],
            "architecture": m["architecture"], "parameters": m["parameters"], "experiment_id": m["experiment_id"],
            "condition": m["condition"], "seed": m["seed"], "epochs": m["epochs"], "optimizer": m["optimizer"],
            "learning_rate": m["learning_rate"], "weight_decay": m["weight_decay"], "fusion": m["fusion"],
            "checkpoint_selection": m["checkpoint_selection"], "training_date": m["training_date"],
            "preprocessing_version": m["preprocessing_version"],
            "parity_with_frozen_artifact": {"artifact": par["artifact"], "exact_match": par["exact_match"],
                                            "max_abs_diff": par["max_abs_diff"]},
            "validation_metrics": m["validation_metrics"],
            "test_metrics_evaluation_only": m["test_metrics_evaluation_only"], "device": str(pkg.device)}


def limitations(pkg: Package) -> list[str]:
    ev = pkg.evaluation
    j = ev["multimodal_joint_head"]
    nc = pkg.labels["n_classes"]
    out = [
        f"Trained from scratch on {pkg.metadata['dataset']} only (train split). Held-out joint accuracy on "
        f"unseen people: {j['test_evaluation_only']['accuracy']:.1%} (test, evaluation-only); chance is {1/nc:.1%}. "
        "Accuracy on your own recordings is unknown and is not measured by anything shown here.",
        f"Confidence is not correctness. Calibration error (ECE) of the joint head on validation is "
        f"{j['validation']['ece_15bin']:.3f}; a high top probability does not make a prediction reliable.",
        "The distribution-shift detector (Mahalanobis distance on encoder representations) is fitted on training "
        "data with its threshold set on validation data so about 5% of in-distribution inputs are flagged. Its "
        "ability to catch out-of-distribution inputs is unknown: passing it is not evidence of reliability.",
    ]
    for m, v in ev["unimodal_frozen_encoder_probes"].items():
        acc = v["test_evaluation_only"]["accuracy"]
        if acc < 2.0 / nc:
            out.append(f"The {m}-only probe is weak: {acc:.1%} test accuracy versus {1/nc:.1%} chance. "
                       f"{m.capitalize()}-only predictions carry little information.")
    for name, o in ev["ood_detector"].items():
        if o["test_flag_rate_evaluation_only"] > 0.15:
            out.append(f"The {name} shift detector flags {o['test_flag_rate_evaluation_only']:.0%} of in-distribution "
                       "test inputs (different subjects), so it is shown only as a diagnostic and is not used for the status.")
    if pkg.dataset == "cremad":
        out += ["CREMA-D contains acted emotions from 91 actors reading 12 scripted sentences in a studio. Real, "
                "spontaneous emotion differs; this is not an emotion-recognition system.",
                "Speech and face checks are heuristics: they reject silence, tones, steady noise and frames without a "
                "frontal face in the centre crop, but cannot reliably tell speech from singing or rhythmic music."]
    else:
        out += ["Action classes are reported by UTD-MHAD action ID (a1-a27); the action names are not stored in the "
                "project and are not reconstructed.",
                "UTD-MHAD inputs must be Kinect-style depth maps (mm) and a 6-axis IMU signal (g, deg/s) from a "
                "sensor worn as in the dataset. Other sensors, placements or units are a distribution shift."]
    out.append("This baseline is separate from the paper's forward-only, gradient-only and ordinary scaling "
               "conditions; no scaling intervention is applied at inference.")
    return out
