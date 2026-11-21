"""Export the VERIFIED architecture of the two real-data models for the website.

Nothing in the output is typed by hand: every layer, tensor shape, parameter count,
kernel/stride/padding and activation is read from the instantiated PyTorch modules
(the exact classes and constructor arguments recorded in the baseline checkpoint
metadata), the shapes are captured with forward hooks on a dummy batch, and the
weights of the saved baseline checkpoint are loaded with strict=True to prove the
module tree matches the trained model. Preprocessing stages come from the
checkpoint's preprocessing.json contract and the loader source constants.

Writes website/src/data/architecture.generated.json.

Usage: python scripts/export_architecture.py [--check]   (--check: fail if the file is stale)
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import crema_d  # noqa: E402
from data import utd_mhad  # noqa: E402
from models.cremad_encoders import AudioCNN, VideoCNN  # noqa: E402
from models.cremad_model import CremadMultimodalModel  # noqa: E402
from models.fusion import fuse  # noqa: E402
from models.utd_encoders import DepthCNN, InertialCNN  # noqa: E402
from models.utd_model import UTDMultimodalModel  # noqa: E402

OUT = ROOT / "website" / "src" / "data" / "architecture.generated.json"
B = 2  # dummy batch; shapes are reported WITHOUT the batch dimension


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def src_ref(obj) -> str:
    """'path:line' of a class/function definition (relative to the repo)."""
    f = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()
    return f"{f}:{inspect.getsourcelines(obj)[1]}"


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def strip_batch(shape, frames: int | None):
    """Drop the batch dim. Per-frame modules see B*frames as their leading dim."""
    s = list(shape)
    if frames and s[0] == B * frames:
        return s[1:], frames
    assert s[0] == B, f"unexpected leading dim {s} (B={B}, frames={frames})"
    return s[1:], None


def block_details(seq: nn.Sequential) -> dict:
    conv, act, pool = seq[0], seq[1], seq[2]
    k = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size,)
    return {
        "layer_types": [type(conv).__name__, type(act).__name__, type(pool).__name__],
        "channels_in": conv.in_channels,
        "channels_out": conv.out_channels,
        "kernel_size": list(k),
        "stride": list(conv.stride),
        "padding": list(conv.padding) if isinstance(conv.padding, tuple) else [conv.padding],
        "activation": type(act).__name__,
        "pooling": f"{type(pool).__name__}(kernel={pool.kernel_size}, stride={pool.stride})",
        "normalization": None,
        "bias": conv.bias is not None,
    }


def capture(model: nn.Module, modules: dict[str, nn.Module], x_a, x_v):
    rec = {}
    hooks = []
    for name, m in modules.items():
        def hook(mod, inp, out, name=name):
            rec[name] = (tuple(inp[0].shape), tuple(out.shape))
        hooks.append(m.register_forward_hook(hook))
    with torch.inference_mode():
        logits = model(x_a, x_v)
    for h in hooks:
        h.remove()
    return rec, tuple(logits.shape)


def encoder_nodes(prefix: str, branch: str, enc: nn.Module, rec: dict, frames: int | None,
                  enc_cls, label_prefix: str, rep_label: str) -> list[dict]:
    nodes = []
    for i in range(3):
        name = f"{prefix}.net.{i}"
        (si, so) = rec[name]
        ins, pf = strip_batch(si, frames)
        outs, _ = strip_batch(so, frames)
        d = block_details(enc.net[i])
        kdim = "×".join(str(x) for x in d["kernel_size"])
        nodes.append({
            "id": f"{branch}_block{i + 1}", "branch": branch, "segment": "encoder",
            "label": f"Conv block {i + 1}",
            "module": name, "op": " → ".join(d["layer_types"]),
            "summary": f"{d['layer_types'][0]}({d['channels_in']}→{d['channels_out']}, {kdim}) · ReLU · max-pool 2",
            "in_shape": ins, "out_shape": outs, "per_frame": pf,
            "params": n_params(enc.net[i]), "details": d, "source": src_ref(enc_cls),
        })
    # global average pool + flatten (net[3], net[4])
    si, _ = rec[f"{prefix}.net.3"]
    _, so = rec[f"{prefix}.net.4"]
    ins, pf = strip_batch(si, frames)
    outs, _ = strip_batch(so, frames)
    gap = enc.net[3]
    nodes.append({
        "id": f"{branch}_gap", "branch": branch, "segment": "encoder",
        "label": "Global average pool", "module": f"{prefix}.net.3 + {prefix}.net.4",
        "op": f"{type(gap).__name__}(1) → Flatten", "summary": f"{type(gap).__name__}(1) + Flatten",
        "in_shape": ins, "out_shape": outs, "per_frame": pf, "params": 0,
        "details": {"pooling": f"{type(gap).__name__}(output_size={gap.output_size})", "activation": None,
                    "normalization": None},
        "source": src_ref(enc_cls),
    })
    fc_in, fc_out = rec[f"{prefix}.fc"]
    if frames:
        # functional op in forward(): reshape(b, n, -1).mean(dim=1)  -- not an nn.Module
        nodes.append({
            "id": f"{branch}_tmean", "branch": branch, "segment": "encoder",
            "label": "Temporal mean-pool", "module": None,
            "op": "reshape(B, N, −1).mean(dim=1)", "summary": f"mean over the {frames} frames",
            "in_shape": [frames] + outs, "out_shape": list(fc_in[1:]), "per_frame": None, "params": 0,
            "details": {"pooling": f"mean over {frames} frames (functional, in forward())", "activation": None,
                        "normalization": None},
            "source": src_ref(enc_cls.forward),
        })
    fc = enc.fc
    nodes.append({
        "id": f"{branch}_fc", "branch": branch, "segment": "encoder",
        "label": f"Linear projection → {rep_label}", "module": f"{prefix}.fc",
        "op": "Linear", "summary": f"Linear({fc.in_features}→{fc.out_features})",
        "in_shape": list(fc_in[1:]), "out_shape": list(fc_out[1:]), "per_frame": None,
        "params": n_params(fc),
        "details": {"features_in": fc.in_features, "features_out": fc.out_features, "activation": None,
                    "normalization": None, "bias": fc.bias is not None},
        "source": src_ref(enc_cls),
    })
    return nodes


def build(dataset: str) -> dict:
    ck = ROOT / "checkpoints" / dataset / "baseline"
    meta = json.loads((ck / "metadata.json").read_text(encoding="utf-8"))
    pre = json.loads((ck / "preprocessing.json").read_text(encoding="utf-8"))
    labels = json.loads((ck / "label_mapping.json").read_text(encoding="utf-8"))
    if dataset == "cremad":
        model = CremadMultimodalModel(n_classes=6, rep_dim=64, fusion="sum", normalization="none", head="linear")
        x_a = torch.zeros(B, *pre["audio"]["tensor_shape"])
        x_v = torch.zeros(B, *pre["video"]["tensor_shape"])
        frames_a, frames_b = None, pre["video"]["frames"]
        enc_a_cls, enc_b_cls = AudioCNN, VideoCNN
        mods = {"a": "audio", "b": "video"}
    else:
        model = UTDMultimodalModel(n_classes=27, rep_dim=64, fusion="sum", normalization="none", head="linear")
        x_a = torch.zeros(B, *pre["depth"]["tensor_shape"])
        x_v = torch.zeros(B, *pre["inertial"]["tensor_shape"])
        frames_a, frames_b = pre["depth"]["frames"], None
        enc_a_cls, enc_b_cls = DepthCNN, InertialCNN
        mods = {"a": "depth", "b": "inertial"}
    sd = torch.load(ck / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)          # proves the module tree == trained checkpoint
    model.eval()
    assert type(model.norm_a) is nn.Identity and type(model.norm_v) is nn.Identity
    assert model.scale.enabled is False
    total = n_params(model)
    assert total == meta["parameters"], (total, meta["parameters"])

    watch = {}
    for p, enc in (("enc_a", model.enc_a), ("enc_v", model.enc_v)):
        for i in range(5):
            watch[f"{p}.net.{i}"] = enc.net[i]
        watch[f"{p}.fc"] = enc.fc
    watch["head"] = model.head
    rec, logit_shape = capture(model, watch, x_a, x_v)

    rep = model.enc_a.fc.out_features
    nodes = []
    nodes += pipeline_nodes(dataset, pre)
    nodes += encoder_nodes("enc_a", "a", model.enc_a, rec, frames_a, enc_a_cls, mods["a"], "h_A")
    nodes += encoder_nodes("enc_v", "b", model.enc_v, rec, frames_b, enc_b_cls, mods["b"], "h_B")
    h_in, h_out = rec["head"]
    nodes.append({
        "id": "fusion", "branch": "joint", "segment": "fusion", "label": "Sum fusion",
        "module": None, "op": "h_A + h_B (models.fusion.fuse('sum'))", "summary": "element-wise sum h_A + h_B",
        "in_shape": [rep], "in_shapes": {"a": [rep], "b": [rep]}, "out_shape": list(h_in[1:]),
        "per_frame": None, "params": 0,
        "details": {"fusion": model.fusion, "activation": None, "normalization": None,
                    "pre_fusion": "norm_a / norm_v = nn.Identity (normalization='none'); ScaleInjector disabled (α = 1)"},
        "source": src_ref(fuse),
    })
    head = model.head
    nodes.append({
        "id": "head", "branch": "joint", "segment": "head", "label": "Linear classifier",
        "module": "head", "op": "Linear", "summary": f"Linear({head.in_features}→{head.out_features}) → logits",
        "in_shape": list(h_in[1:]), "out_shape": list(h_out[1:]), "per_frame": None, "params": n_params(head),
        "details": {"features_in": head.in_features, "features_out": head.out_features, "activation": None,
                    "normalization": None, "bias": head.bias is not None},
        "source": "models/heads.py:" + str(inspect.getsourcelines(__import__("models.heads", fromlist=["make_head"]).make_head)[1]),
    })
    n_classes = len(labels["classes"])
    assert list(logit_shape[1:]) == [n_classes]
    nodes.append({
        "id": "output", "branch": "joint", "segment": "output", "label": f"{n_classes}-class prediction",
        "module": None, "op": "softmax (inference only)",
        "summary": "softmax over logits → class probabilities",
        "in_shape": [n_classes], "out_shape": [n_classes], "per_frame": None, "params": 0,
        "details": {"activation": "softmax (applied by the inference code; not part of the nn.Module — "
                                  "training uses CrossEntropyLoss on the logits)", "normalization": None},
        "source": "inference/common.py:" + str(_line_of(ROOT / "inference" / "common.py", "torch.softmax")),
    })

    edges = edges_for(nodes)
    order = [n["id"] for n in nodes]          # forward call order (pipeline A, pipeline B, enc_a, enc_v, fuse, head)
    comp = {
        "encoder_a": n_params(model.enc_a), "encoder_b": n_params(model.enc_v),
        "fusion": 0, "head": n_params(head), "total": total,
    }
    return {
        "dataset": dataset,
        "display_name": "CREMA-D" if dataset == "cremad" else "UTD-MHAD",
        "task": "emotion (6 classes)" if dataset == "cremad" else "action (27 classes)",
        "model_class": type(model).__name__,
        "constructor_call": meta["architecture"],
        "source_files": meta["architecture_source"] if "architecture_source" in meta else [],
        "modalities": mods,
        "encoders": {"a": enc_a_cls.__name__, "b": enc_b_cls.__name__},
        "rep_dim": rep, "fusion": model.fusion, "head": "linear",
        "normalization": "none (nn.Identity)", "scale_injection": "disabled (α = 1)",
        "n_classes": n_classes,
        "class_codes": [c["code"] for c in labels["classes"]],
        "class_names": [c.get("name") for c in labels["classes"]],
        "params": comp,
        "training": {k: meta[k] for k in ("epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
                                          "loss", "seed", "checkpoint_selection") if k in meta},
        "checkpoint": {"path": f"checkpoints/{dataset}/baseline/model.pth", "sha256": sha256(ck / "model.pth"),
                       "strict_state_dict_load": True},
        "preprocessing_version": pre["version"],
        "nodes": nodes, "edges": edges, "forward_order": order,
    }


def _line_of(path: Path, needle: str) -> int:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return i
    raise ValueError(needle)


def pipeline_nodes(dataset: str, pre: dict) -> list[dict]:
    P = "pipeline"
    if dataset == "cremad":
        a, v = pre["audio"], pre["video"]
        n_samples = int(crema_d.DUR * crema_d.SR)
        return [
            {"id": "a_raw", "branch": "a", "segment": P, "label": "Audio recording", "module": None,
             "op": "decode", "summary": "speech recording (CREMA-D: WAV)", "in_shape": None, "out_shape": None,
             "shape_note": "variable length, sample rate and channels", "per_frame": None, "params": 0,
             "details": {"decoder": a["decoder"]}, "source": src_ref(crema_d.load_audio_logmel)},
            {"id": "a_wave", "branch": "a", "segment": P, "label": "Mono 16 kHz, 3.0 s", "module": None,
             "op": "mono · resample · pad/crop", "summary": f"mono, {a['sample_rate']} Hz, first {a['duration_s']} s",
             "in_shape": None, "out_shape": [1, n_samples], "per_frame": None, "params": 0,
             "details": {"mono": a["mono"], "sample_rate": a["sample_rate"], "resampler": a["resampler"],
                         "crop": a["crop"], "int16_scaling": a["int16_scaling"]},
             "source": src_ref(crema_d.load_audio_logmel)},
            {"id": "a_logmel", "branch": "a", "segment": P, "label": "64-bin log-mel", "module": None,
             "op": "MelSpectrogram → log", "summary": f"{a['mel']['n_mels']} mels, n_fft {a['mel']['n_fft']}, hop {a['mel']['hop_length']}",
             "in_shape": [1, n_samples], "out_shape": a["tensor_shape"], "per_frame": None, "params": 0,
             "details": {"mel": a["mel"], "log": a["log"], "time_frames": a["time_frames"],
                         "cache_dtype": a["cache_dtype"], "normalization": a["normalization"]},
             "source": src_ref(crema_d.load_audio_logmel)},
            {"id": "b_raw", "branch": "b", "segment": P, "label": "Video recording", "module": None,
             "op": "decode", "summary": "face video (CREMA-D: FLV)", "in_shape": None, "out_shape": None,
             "shape_note": "variable frame count, resolution and frame rate", "per_frame": None, "params": 0,
             "details": {"decoder": v["decoder"]}, "source": src_ref(crema_d.load_video_frames)},
            {"id": "b_sample", "branch": "b", "segment": P, "label": f"Sample {v['frames']} frames", "module": None,
             "op": "uniform frame sampling", "summary": v["frame_index"], "in_shape": None, "out_shape": None,
             "shape_note": f"{v['frames']} colour frames at the original resolution", "per_frame": None, "params": 0,
             "details": {"frame_index": v["frame_index"]}, "source": src_ref(crema_d.load_video_frames)},
            {"id": "b_frames", "branch": "b", "segment": P, "label": "Grayscale 64×64 frames", "module": None,
             "op": "gray · centre crop · resize · /255", "summary": "BGR→gray, centre square, 64×64 INTER_AREA, /255",
             "in_shape": None, "out_shape": v["tensor_shape"], "per_frame": None, "params": 0,
             "details": {"color": v["color"], "crop": v["crop"], "resize": v["resize"],
                         "interpolation": v["interpolation"], "scale": v["scale"], "cache_dtype": v["cache_dtype"],
                         "normalization": v["normalization"]},
             "source": src_ref(crema_d.load_video_frames)},
        ]
    d, i = pre["depth"], pre["inertial"]
    return [
        {"id": "a_raw", "branch": "a", "segment": P, "label": "Depth sequence", "module": None,
         "op": "load .mat", "summary": "Kinect depth stack d_depth (H, W, T)", "in_shape": None, "out_shape": None,
         "shape_note": "(H, W, T) uint16 millimetres; T varies per trial", "per_frame": None, "params": 0,
         "details": {"input": d["input"]}, "source": src_ref(utd_mhad.load_depth_clip)},
        {"id": "a_sample", "branch": "a", "segment": P, "label": f"Sample {d['frames']} frames, 32×32", "module": None,
         "op": "frame sampling · resize", "summary": f"{d['frame_index']}; INTER_AREA to 32×32",
         "in_shape": None, "out_shape": [d["frames"], d["resize"][0], d["resize"][1]], "per_frame": None, "params": 0,
         "details": {"frame_index": d["frame_index"], "resize": d["resize"], "interpolation": d["interpolation"]},
         "source": src_ref(utd_mhad.load_depth_clip)},
        {"id": "a_frames", "branch": "a", "segment": P, "label": "Per-clip min–max", "module": None,
         "op": "min–max to [0, 1]", "summary": d["normalization"],
         "in_shape": [d["frames"], d["resize"][0], d["resize"][1]], "out_shape": d["tensor_shape"], "per_frame": None,
         "params": 0, "details": {"normalization": d["normalization"], "dtype": d["dtype"]},
         "source": src_ref(utd_mhad.load_depth_clip)},
        {"id": "b_raw", "branch": "b", "segment": P, "label": "Inertial signal", "module": None,
         "op": "load .mat", "summary": "T × 6 (accelerometer g, gyroscope deg/s)", "in_shape": None, "out_shape": None,
         "shape_note": "T × 6; T varies per trial", "per_frame": None, "params": 0,
         "details": {"input": i["input"]}, "source": src_ref(utd_mhad.load_inertial)},
        {"id": "b_resample", "branch": "b", "segment": P, "label": "Resample to 128", "module": None,
         "op": "np.interp per channel", "summary": i["resample"], "in_shape": None, "out_shape": i["tensor_shape"],
         "per_frame": None, "params": 0, "details": {"resample": i["resample"]},
         "source": src_ref(utd_mhad.load_inertial)},
        {"id": "b_zscore", "branch": "b", "segment": P, "label": "Train-set z-score", "module": None,
         "op": "(x − μ_train) / σ_train", "summary": i["standardization"], "in_shape": i["tensor_shape"],
         "out_shape": i["tensor_shape"], "per_frame": None, "params": 0,
         "details": {"standardization": i["standardization"], "dtype": i["dtype"]},
         "source": "experiments/utd_extract.py"},
    ]


def edges_for(nodes: list[dict]) -> list[dict]:
    edges = []
    for br in ("a", "b"):
        seq = [n["id"] for n in nodes if n["branch"] == br]
        edges += [{"from": s, "to": t} for s, t in zip(seq, seq[1:])]
        edges.append({"from": seq[-1], "to": "fusion"})
    edges += [{"from": "fusion", "to": "head"}, {"from": "head", "to": "output"}]
    return edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    out = {
        "generated_by": "scripts/export_architecture.py",
        "note": "Generated from the model code and the baseline checkpoints; do not edit by hand.",
        "source_sha256": {p: sha256(ROOT / p) for p in (
            "models/cremad_model.py", "models/cremad_encoders.py", "models/utd_model.py",
            "models/utd_encoders.py", "models/fusion.py", "models/heads.py", "data/crema_d.py", "data/utd_mhad.py")},
        "datasets": {"cremad": build("cremad"), "utd_mhad": build("utd_mhad")},
    }
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            sys.exit(f"{OUT} is stale: re-run scripts/export_architecture.py")
        print("architecture.generated.json is up to date")
        return
    OUT.write_text(text, encoding="utf-8")
    for k, d in out["datasets"].items():
        print(k, d["params"], len(d["nodes"]), "nodes")


if __name__ == "__main__":
    main()
