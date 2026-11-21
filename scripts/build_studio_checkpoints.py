"""Build the Prediction Studio baseline checkpoints (CREMA-D, UTD-MHAD).

Faithfulness guarantees
  * Training is done by the ORIGINAL experiment functions, unchanged:
      CREMA-D  -> experiments.phase2_cremad_probe.train_one(cfg, feats, device, seed, "baseline")
      UTD-MHAD -> experiments.phase4_utd_decoupled.train(cfg, feats, device, seed, "baseline")
    Each ends by calling the module-level `decoupled(...)` evaluation; we wrap that call
    only to *capture* the trained model. Architecture, optimizer, epochs, batch size,
    seed, split and the fixed-final-epoch policy are therefore exactly the documented ones.
  * Parity: the metrics returned by that original evaluation are compared with the frozen
    seed-0 rows in results/ (which this script never writes to).
  * Train-only fitting: unimodal probes and the OOD reference statistics use TRAIN
    representations; the OOD threshold uses VALIDATION; TEST is evaluation-only.

Usage:  python scripts/build_studio_checkpoints.py --dataset {cremad,utd_mhad,all} [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from inference.ood_detection import MahalanobisDetector
from inference.probes import fit_linear_probe, probe_logits, PROBE_LR, PROBE_STEPS, PROBE_WEIGHT_DECAY
from msc.device import device_report, get_device
from msc.runlog import git_hash, load_config
from msc.seed import set_seed

SEED = 0                      # first documented seed (configs: seeds [0,1,2,3,4])
CKPT_ROOT = ROOT / "checkpoints"
EVAL_BS = 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def macro_f1(pred: np.ndarray, y: np.ndarray, nc: int) -> float:
    f1s = []
    for c in range(nc):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


@torch.inference_mode()
def encode_all(model, a, v):
    model.eval(); ha, hv, lg = [], [], []
    for i in range(0, a.shape[0], EVAL_BS):
        h_a, h_v = model.encode(a[i:i + EVAL_BS], v[i:i + EVAL_BS])
        ha.append(h_a); hv.append(h_v); lg.append(model(a[i:i + EVAL_BS], v[i:i + EVAL_BS]))
    return torch.cat(ha), torch.cat(hv), torch.cat(lg)


def classification_report(logits: torch.Tensor, y: torch.Tensor, nc: int) -> dict:
    p = torch.softmax(logits.float(), 1).cpu().numpy(); yy = y.cpu().numpy(); pred = p.argmax(1)
    return {"accuracy": float((pred == yy).mean()), "macro_f1": macro_f1(pred, yy, nc),
            "ece_15bin": ece(p, yy), "n": int(len(yy)),
            "mean_max_softmax": float(p.max(1).mean())}


def parity(returned: dict, frozen_row: pd.Series, keys: list[str]) -> dict:
    out = {}
    for k in keys:
        r, f = float(returned[k]), float(frozen_row[k])
        out[k] = {"retrained": r, "frozen": f, "abs_diff": abs(r - f)}
    out["max_abs_diff"] = max(v["abs_diff"] for v in out.values() if isinstance(v, dict))
    out["exact_match"] = out["max_abs_diff"] < 1e-6
    return out


def capture_and_train(module, train_fn_name: str, args: tuple):
    """Run module.<train_fn_name>(*args) unchanged; capture the model passed to decoupled()."""
    captured = {}
    original = module.decoupled

    def wrapper(model, *a, **k):
        captured["model"] = model
        return original(model, *a, **k)

    module.decoupled = wrapper
    try:
        result = getattr(module, train_fn_name)(*args)
    finally:
        module.decoupled = original
    if result.get("diverged"):
        raise RuntimeError(f"original training diverged: {result}")
    return captured["model"], result


def write_package(out: Path, *, model, cfg: dict, metadata: dict, labels: dict, preprocessing: dict,
                  evaluation: dict, probes: dict, detectors: dict, norm_stats: dict | None, force: bool):
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force to rebuild (deterministic)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model.pth")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "label_mapping.json").write_text(json.dumps(labels, indent=2))
    (out / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    np.savez(out / "probes.npz", **{f"{m}_{k}": v for m, p in probes.items() for k, v in p.items()})
    ood = {}
    for name, det in detectors.items():
        ood.update(det.to_npz_dict(name))
    np.savez(out / "ood_detector.npz", **ood)
    if norm_stats is not None:
        np.savez(out / "normalization_stats.npz", **norm_stats)
    metadata["checkpoint_hash"] = {"algorithm": "sha256", "model.pth": sha256(out / "model.pth")}
    metadata["package_files_sha256"] = {p.name: sha256(p) for p in sorted(out.iterdir())}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def fit_unimodal_and_ood(h_tr_a, h_tr_v, y_tr, h_va_a, h_va_v, y_va, h_te_a, h_te_v, y_te, nc, device, names):
    set_seed(SEED, True)   # probe init is the only stochastic step here
    probes = {names[0]: fit_linear_probe(h_tr_a, y_tr, nc, device),
              names[1]: fit_linear_probe(h_tr_v, y_tr, nc, device)}
    uni_eval = {}
    for m, (hva, hte) in {names[0]: (h_va_a, h_te_a), names[1]: (h_va_v, h_te_v)}.items():
        uni_eval[m] = {"validation": classification_report(probe_logits(probes[m], hva), y_va, nc),
                       "test_evaluation_only": classification_report(probe_logits(probes[m], hte), y_te, nc)}
    np_ = lambda t: t.detach().cpu().numpy()
    ytr = np_(y_tr)
    detectors = {
        names[0]: MahalanobisDetector.fit(np_(h_tr_a), ytr, np_(h_va_a)),
        names[1]: MahalanobisDetector.fit(np_(h_tr_v), ytr, np_(h_va_v)),
        "joint": MahalanobisDetector.fit(np.concatenate([np_(h_tr_a), np_(h_tr_v)], 1), ytr,
                                         np.concatenate([np_(h_va_a), np_(h_va_v)], 1)),
    }
    test_feats = {names[0]: np_(h_te_a), names[1]: np_(h_te_v),
                  "joint": np.concatenate([np_(h_te_a), np_(h_te_v)], 1)}
    ood_eval = {k: {"fit_on": "train", "threshold_from": "validation 95th percentile",
                    "threshold": d.threshold, "validation_score_quantiles": d.val_scores_q,
                    "validation_flag_rate": 0.05,
                    "test_flag_rate_evaluation_only": float((d.score(test_feats[k]) > d.threshold).mean())}
                for k, d in detectors.items()}
    return probes, uni_eval, detectors, ood_eval


def build_cremad(force: bool):
    import experiments.phase2_cremad_probe as P2
    cfg = load_config(ROOT / "configs" / "cremad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    feats = P2.load_features(device)
    nc = cfg["data"]["n_classes"]
    print("[cremad] training with ORIGINAL phase2_cremad_probe.train_one(seed=0, arm='baseline') ...")
    model, returned = capture_and_train(P2, "train_one", (cfg, feats, device, SEED, "baseline"))
    frozen = pd.read_csv(ROOT / "results/phase2_mechanism/phase2_cremad_probe.csv")
    frozen_row = frozen[(frozen.arm == "baseline") & (frozen.seed == SEED)].iloc[0]
    par = parity(returned, frozen_row, ["probe_audio", "probe_video", "acc_audio", "acc_video",
                                        "dominance", "meanabl_dominance", "shapley_audio"])
    print(f"[cremad] parity vs frozen seed-0 row: max |diff| = {par['max_abs_diff']:.2e}  exact={par['exact_match']}")

    (a_tr, v_tr, y_tr), (a_va, v_va, y_va), (a_te, v_te, y_te) = feats["train"], feats["val"], feats["test"]
    ha_tr, hv_tr, _ = encode_all(model, a_tr, v_tr)
    ha_va, hv_va, lg_va = encode_all(model, a_va, v_va)
    ha_te, hv_te, lg_te = encode_all(model, a_te, v_te)
    probes, uni_eval, detectors, ood_eval = fit_unimodal_and_ood(
        ha_tr, hv_tr, y_tr, ha_va, hv_va, y_va, ha_te, hv_te, y_te, nc, device, ("audio", "video"))

    labels = {"dataset": "CREMA-D", "n_classes": 6, "source": "data/cremad_manifest.py::EMO",
              "classes": [{"index": i, "code": c, "name": n} for i, (c, n) in enumerate(
                  [("ANG", "anger"), ("DIS", "disgust"), ("FEA", "fear"), ("HAP", "happy"),
                   ("NEU", "neutral"), ("SAD", "sad")])]}
    pre = {"version": "cremad-train-identical-v1",
           "source_files_sha256": {"data/crema_d.py": sha256(ROOT / "data/crema_d.py"),
                                   "experiments/cremad_extract.py": sha256(ROOT / "experiments/cremad_extract.py")},
           "audio": {"function": "data.crema_d.load_audio_logmel", "decoder": "scipy.io.wavfile (WAV); "
                     "other containers are first decoded by PyAV to 16-bit PCM WAV",
                     "mono": "mean over channels", "int16_scaling": "divide by 32768 if max|x|>1",
                     "sample_rate": 16000, "resampler": "torchaudio.functional.resample (defaults)",
                     "duration_s": 3.0, "crop": "first 3.0 s; zero-pad at end if shorter",
                     "mel": {"n_fft": 1024, "hop_length": 256, "n_mels": 64, "win_length": 1024,
                             "power": 2.0, "center": True},
                     "log": "log(mel + 1e-6)", "time_frames": 188, "cache_dtype": "float16",
                     "tensor_shape": [1, 64, 188], "normalization": "none", "augmentation": "none"},
           "video": {"function": "data.crema_d.load_video_frames", "decoder": "cv2.VideoCapture (FFmpeg)",
                     "frames": 8, "frame_index": "np.linspace(0, n-1, 8).astype(int) over all frames",
                     "color": "BGR->GRAY", "crop": "centre square min(h,w)",
                     "resize": [64, 64], "interpolation": "INTER_AREA", "scale": "/255",
                     "cache_dtype": "float16", "tensor_shape": [8, 1, 64, 64],
                     "normalization": "none", "augmentation": "none"}}
    evaluation = {
        "note": "Test-split numbers are evaluation-only; nothing was tuned on test.",
        "parity_with_frozen_artifact": {"artifact": "results/phase2_mechanism/phase2_cremad_probe.csv",
                                        "row": {"seed": SEED, "arm": "baseline"}, **par},
        "multimodal_joint_head": {"validation": classification_report(lg_va, y_va, nc),
                                  "test_evaluation_only": classification_report(lg_te, y_te, nc)},
        "unimodal_frozen_encoder_probes": uni_eval, "ood_detector": ood_eval}
    meta = {"architecture": "CremadMultimodalModel(n_classes=6, rep_dim=64, fusion='sum', "
                            "normalization='none', head='linear')",
            "architecture_source": ["models/cremad_model.py", "models/cremad_encoders.py",
                                    "models/fusion.py", "models/heads.py"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "dataset": "CREMA-D (actor-disjoint split 64/9/18 actors, 5234/738/1470 clips)",
            "experiment_id": "phase2_cremad_probe.train_one / arm=baseline (no scaling)",
            "condition": "baseline only -- NOT the forward-only, gradient-only or ordinary arms",
            "seed": SEED, "epochs": cfg["train"]["epochs"], "batch_size": cfg["train"]["batch_size"],
            "optimizer": "Adam", "learning_rate": cfg["train"]["lr"], "weight_decay": cfg["train"]["weight_decay"],
            "loss": "CrossEntropyLoss", "fusion": cfg["model"]["fusion"],
            "checkpoint_selection": "fixed final epoch (documented policy; no early stopping, no validation selection)",
            "trained_on": "train split only",
            "training_date": dt.datetime.now().isoformat(timespec="seconds"),
            "validation_metrics": evaluation["multimodal_joint_head"]["validation"],
            "test_metrics_evaluation_only": evaluation["multimodal_joint_head"]["test_evaluation_only"],
            "preprocessing_version": pre["version"], "git_commit": git_hash(),
            "torch_version": torch.__version__, "device": str(device),
            "probe_protocol": {"source": "analysis/probes.py", "steps": PROBE_STEPS, "lr": PROBE_LR,
                               "weight_decay": PROBE_WEIGHT_DECAY, "fit_on": "train representations"},
            "research_prototype": True}
    cfg_out = {**cfg, "studio": {"seed": SEED, "arm": "baseline", "source_config": "configs/cremad.yaml"}}
    meta = write_package(CKPT_ROOT / "cremad" / "baseline", model=model, cfg=cfg_out, metadata=meta,
                         labels=labels, preprocessing=pre, evaluation=evaluation, probes=probes,
                         detectors=detectors, norm_stats=None, force=force)
    print("[cremad] saved; model sha256", meta["checkpoint_hash"]["model.pth"][:16])
    return evaluation


def build_utd(force: bool):
    import experiments.phase4_utd_decoupled as U
    cfg = load_config(ROOT / "configs" / "utd_mhad.yaml")
    device = get_device(cfg["device"]); print("Device:", device_report(device))
    cache = ROOT / cfg["data"]["cache_dir"]
    feats = U.load_utd(cache, device)                       # train/test, exactly as the experiment
    nc = cfg["data"]["n_classes"]
    print("[utd_mhad] training with ORIGINAL phase4_utd_decoupled.train(seed=0, arm='baseline') ...")
    model, returned = capture_and_train(U, "train", (cfg, feats, device, SEED, "baseline"))
    frozen = pd.read_csv(ROOT / "results/phase4_baselines/phase4_utd.csv")
    frozen_row = frozen[(frozen.arm == "baseline") & (frozen.seed == SEED)].iloc[0]
    par = parity(returned, frozen_row, ["probe_depth", "probe_inertial", "acc_depth", "acc_inertial",
                                        "dominance", "meanabl_dominance", "shapley_scaled"])
    print(f"[utd_mhad] parity vs frozen seed-0 row: max |diff| = {par['max_abs_diff']:.2e}  exact={par['exact_match']}")

    a_va = torch.from_numpy(np.load(cache / "val_depth.npy")).float().to(device)
    v_va = torch.from_numpy(np.load(cache / "val_inertial.npy")).float().to(device)
    y_va = torch.from_numpy(np.load(cache / "val_y.npy")).long().to(device)
    (a_tr, v_tr, y_tr), (a_te, v_te, y_te) = feats["train"], feats["test"]
    ha_tr, hv_tr, _ = encode_all(model, a_tr, v_tr)
    ha_va, hv_va, lg_va = encode_all(model, a_va, v_va)
    ha_te, hv_te, lg_te = encode_all(model, a_te, v_te)
    probes, uni_eval, detectors, ood_eval = fit_unimodal_and_ood(
        ha_tr, hv_tr, y_tr, ha_va, hv_va, y_va, ha_te, hv_te, y_te, nc, device, ("depth", "inertial"))

    stats = json.loads((cache / "extract_stats.json").read_text())
    labels = {"dataset": "UTD-MHAD", "n_classes": 27, "source": "data/utd_manifest.py (label = action_id - 1)",
              "names_available": False,
              "names_note": "Action names are not stored in the repository or the raw dataset folder and "
                            "could not be verified from the official page; classes are reported by action ID.",
              "classes": [{"index": i, "code": f"a{i + 1}", "action_id": i + 1, "name": None} for i in range(27)]}
    pre = {"version": "utd-mhad-train-identical-v1",
           "source_files_sha256": {"data/utd_mhad.py": sha256(ROOT / "data/utd_mhad.py"),
                                   "experiments/utd_extract.py": sha256(ROOT / "experiments/utd_extract.py")},
           "depth": {"function": "data.utd_mhad.load_depth_clip", "input": "Kinect depth map stack, "
                     "UTD .mat (variable d_depth, (H,W,T) uint16 mm) or .npy (T,H,W)",
                     "frames": 16, "frame_index": "np.linspace(0, T-1, 16).round()",
                     "resize": [32, 32], "interpolation": "INTER_AREA on raw depth",
                     "normalization": "per-clip min-max over the 16 frames -> [0,1]",
                     "dtype": "float16", "tensor_shape": [16, 1, 32, 32]},
           "inertial": {"function": "data.utd_mhad.load_inertial + train z-score (experiments/utd_extract.py)",
                        "input": "T x 6 signal: columns 0-2 accelerometer (g), 3-5 gyroscope (deg/s), "
                                 "UTD .mat (variable d_iner) or .csv",
                        "resample": "np.interp per channel to 128 points over [0, T-1]",
                        "standardization": "per-channel z-score with TRAIN mean/std (normalization_stats.npz)",
                        "dtype": "float16", "tensor_shape": [6, 128]}}
    norm = {"inertial_channel_mean": np.array(stats["inertial_channel_mean"], dtype=np.float64),
            "inertial_channel_std": np.array(stats["inertial_channel_std"], dtype=np.float64)}
    evaluation = {
        "note": "Test-split numbers are evaluation-only; nothing was tuned on test.",
        "parity_with_frozen_artifact": {"artifact": "results/phase4_baselines/phase4_utd.csv",
                                        "row": {"seed": SEED, "arm": "baseline"}, **par},
        "multimodal_joint_head": {"validation": classification_report(lg_va, y_va, nc),
                                  "test_evaluation_only": classification_report(lg_te, y_te, nc)},
        "unimodal_frozen_encoder_probes": uni_eval, "ood_detector": ood_eval}
    meta = {"architecture": "UTDMultimodalModel(n_classes=27, rep_dim=64, fusion='sum', "
                            "normalization='none', head='linear')",
            "architecture_source": ["models/utd_model.py", "models/utd_encoders.py",
                                    "models/fusion.py", "models/heads.py"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "dataset": "UTD-MHAD (subject split train {1,3,5} / val {7} / test {2,4,6,8}; 323/108/430)",
            "experiment_id": "phase4_utd_decoupled.train / arm=baseline (no scaling)",
            "condition": "baseline only -- NOT the forward-only or gradient-only arms",
            "seed": SEED, "epochs": cfg["train"]["epochs"], "batch_size": cfg["train"]["batch_size"],
            "optimizer": "Adam", "learning_rate": cfg["train"]["lr"], "weight_decay": cfg["train"]["weight_decay"],
            "loss": "CrossEntropyLoss", "fusion": cfg["model"]["fusion"],
            "checkpoint_selection": "fixed final epoch (documented policy; no early stopping, no validation selection)",
            "trained_on": "train split only",
            "training_date": dt.datetime.now().isoformat(timespec="seconds"),
            "validation_metrics": evaluation["multimodal_joint_head"]["validation"],
            "test_metrics_evaluation_only": evaluation["multimodal_joint_head"]["test_evaluation_only"],
            "preprocessing_version": pre["version"], "git_commit": git_hash(),
            "torch_version": torch.__version__, "device": str(device),
            "probe_protocol": {"source": "analysis/probes.py", "steps": PROBE_STEPS, "lr": PROBE_LR,
                               "weight_decay": PROBE_WEIGHT_DECAY, "fit_on": "train representations"},
            "research_prototype": True}
    cfg_out = {**cfg, "studio": {"seed": SEED, "arm": "baseline", "source_config": "configs/utd_mhad.yaml"}}
    meta = write_package(CKPT_ROOT / "utd_mhad" / "baseline", model=model, cfg=cfg_out, metadata=meta,
                         labels=labels, preprocessing=pre, evaluation=evaluation, probes=probes,
                         detectors=detectors, norm_stats=norm, force=force)
    print("[utd_mhad] saved; model sha256", meta["checkpoint_hash"]["model.pth"][:16])
    return evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cremad", "utd_mhad", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.dataset in ("cremad", "all"):
        build_cremad(a.force)
    if a.dataset in ("utd_mhad", "all"):
        build_utd(a.force)


if __name__ == "__main__":
    main()
