"""Safe loading of the Studio's own checkpoint packages.

* Only the two fixed package directories under checkpoints/ can be loaded; there is no
  code path that loads a user-supplied model file.
* Every package file's SHA-256 is checked against metadata.json before use.
* Weights load with torch.load(..., weights_only=True) and load_state_dict(strict=True)
  into the research project's own model classes.
* .npz files load with allow_pickle=False.
* Models are put in eval mode with requires_grad disabled; prediction never trains.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from inference import CHECKPOINT_ROOT
from inference.ood_detection import MahalanobisDetector
from models.cremad_model import CremadMultimodalModel
from models.utd_model import UTDMultimodalModel

DATASETS = {
    "cremad": {"cls": CremadMultimodalModel, "modalities": ("audio", "video")},
    "utd_mhad": {"cls": UTDMultimodalModel, "modalities": ("depth", "inertial")},
}


class CheckpointError(RuntimeError):
    pass


@dataclass
class Package:
    dataset: str
    model: torch.nn.Module
    device: torch.device
    metadata: dict
    config: dict
    labels: dict
    preprocessing: dict
    evaluation: dict
    probes: dict
    detectors: dict
    quality_reference: dict | None = None
    norm_stats: dict | None = None
    modalities: tuple = ()
    files_verified: list = field(default_factory=list)

    @property
    def checkpoint_id(self) -> str:
        return f"{self.dataset}/baseline@{self.metadata['checkpoint_hash']['model.pth'][:12]}"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_package(dataset: str, device: torch.device | None = None) -> Package:
    if dataset not in DATASETS:
        raise CheckpointError(f"unknown dataset {dataset!r}")
    pkg = CHECKPOINT_ROOT / dataset / "baseline"
    if not (pkg / "metadata.json").exists():
        raise CheckpointError(f"no checkpoint package at {pkg}; run scripts/build_studio_checkpoints.py")
    meta = json.loads((pkg / "metadata.json").read_text())
    expected = meta.get("package_files_sha256", {})
    if meta["checkpoint_hash"]["model.pth"] != expected.get("model.pth"):
        raise CheckpointError("model.pth hash in metadata is inconsistent")
    verified = []
    for name, digest in expected.items():
        f = pkg / name
        if not f.exists():
            raise CheckpointError(f"package file missing: {name}")
        if _sha256(f) != digest:
            raise CheckpointError(f"hash mismatch for {name}; the package was modified after it was built")
        verified.append(name)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((pkg / "config.yaml").read_text())
    labels = json.loads((pkg / "label_mapping.json").read_text())
    nc = int(cfg["data"]["n_classes"])
    if nc != labels["n_classes"] or nc != len(labels["classes"]):
        raise CheckpointError("label mapping does not match the configured number of classes")
    model = DATASETS[dataset]["cls"](nc, rep_dim=cfg["model"]["rep_dim"], fusion=cfg["model"]["fusion"],
                                     normalization="none", head=cfg["model"]["head"])
    state = torch.load(pkg / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if model.scale.enabled:
        raise CheckpointError("baseline checkpoint must not have scale injection enabled")

    with np.load(pkg / "probes.npz", allow_pickle=False) as z:
        mods = DATASETS[dataset]["modalities"]
        probes = {m: {k: z[f"{m}_{k}"] for k in ("weight", "bias", "mu", "sd")} for m in mods}
    with np.load(pkg / "ood_detector.npz", allow_pickle=False) as z:
        detectors = {k: MahalanobisDetector.from_npz(z, k) for k in (*mods, "joint")}
    norm = None
    if (pkg / "normalization_stats.npz").exists():
        with np.load(pkg / "normalization_stats.npz", allow_pickle=False) as z:
            norm = {k: z[k] for k in z.files}
    qref = json.loads((pkg / "quality_reference.json").read_text()) if (pkg / "quality_reference.json").exists() else None
    return Package(dataset=dataset, model=model, device=device, metadata=meta, config=cfg, labels=labels,
                   preprocessing=json.loads((pkg / "preprocessing.json").read_text()),
                   evaluation=json.loads((pkg / "evaluation.json").read_text()), probes=probes,
                   detectors=detectors, quality_reference=qref, norm_stats=norm,
                   modalities=DATASETS[dataset]["modalities"], files_verified=verified)
