"""Checkpoint packages: integrity, safe loading, label mapping, shapes, reproducibility, parity."""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def test_packages_load_and_verify_every_file(cremad_pkg, utd_pkg):
    for pkg, extra in ((cremad_pkg, "quality_reference.json"), (utd_pkg, "normalization_stats.npz")):
        assert set(pkg.files_verified) >= {"model.pth", "config.yaml", "label_mapping.json", "preprocessing.json",
                                           "evaluation.json", "probes.npz", "ood_detector.npz", extra}
        assert not pkg.model.training
        assert all(not p.requires_grad for p in pkg.model.parameters())
        assert pkg.model.scale.enabled is False                     # baseline: no scaling arm


def test_metadata_fields(cremad_pkg, utd_pkg):
    for pkg in (cremad_pkg, utd_pkg):
        m = pkg.metadata
        for k in ("architecture", "dataset", "experiment_id", "seed", "epochs", "optimizer", "learning_rate",
                  "fusion", "training_date", "validation_metrics", "test_metrics_evaluation_only",
                  "preprocessing_version", "checkpoint_hash"):
            assert k in m, k
        assert m["seed"] == 0 and m["optimizer"] == "Adam" and m["weight_decay"] == 0.0
        assert "fixed final epoch" in m["checkpoint_selection"]
        assert m["condition"].startswith("baseline only")
    assert cremad_pkg.metadata["epochs"] == 30 and utd_pkg.metadata["epochs"] == 60


def test_parity_with_frozen_experiment_is_exact(cremad_pkg, utd_pkg):
    for pkg in (cremad_pkg, utd_pkg):
        par = pkg.evaluation["parity_with_frozen_artifact"]
        assert par["exact_match"] is True and par["max_abs_diff"] < 1e-6


def test_tampered_package_is_rejected(tmp_path, monkeypatch):
    import inference.model_loader as ml
    src = ROOT / "checkpoints" / "cremad" / "baseline"
    dst = tmp_path / "cremad" / "baseline"
    shutil.copytree(src, dst)
    data = bytearray((dst / "probes.npz").read_bytes()); data[-10] ^= 0xFF
    (dst / "probes.npz").write_bytes(bytes(data))
    monkeypatch.setattr(ml, "CHECKPOINT_ROOT", tmp_path)
    with pytest.raises(ml.CheckpointError, match="hash mismatch"):
        ml.load_package("cremad", torch.device("cpu"))


class _Evil:
    def __reduce__(self):
        return (print, ("pickle payload executed",))


def test_weights_only_loading_refuses_arbitrary_pickles(tmp_path):
    p = tmp_path / "evil.pth"
    with open(p, "wb") as f:
        pickle.dump({"x": _Evil()}, f)
    with pytest.raises(Exception):
        torch.load(p, weights_only=True)


def test_label_mappings(cremad_pkg, utd_pkg):
    from data.cremad_manifest import EMO
    codes = [c["code"] for c in cremad_pkg.labels["classes"]]
    assert codes == sorted(EMO, key=EMO.get) == ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    utd = utd_pkg.labels["classes"]
    assert [c["code"] for c in utd] == [f"a{i}" for i in range(1, 28)]
    assert all(c["index"] == c["action_id"] - 1 and c["name"] is None for c in utd)


def test_exact_tensor_shapes(cremad_pkg, utd_pkg):
    with torch.inference_mode():
        d = cremad_pkg.device
        assert cremad_pkg.model(torch.zeros(2, 1, 64, 188, device=d), torch.zeros(2, 8, 1, 64, 64, device=d)).shape == (2, 6)
        d = utd_pkg.device
        assert utd_pkg.model(torch.zeros(2, 16, 1, 32, 32, device=d), torch.zeros(2, 6, 128, device=d)).shape == (2, 27)


def _accuracy(pkg, a, v, y):
    """Exact count-based accuracy (float64), as recorded in evaluation.json."""
    with torch.inference_mode():
        correct = int((pkg.model(a.to(pkg.device), v.to(pkg.device)).argmax(1).cpu() == y).sum())
    return correct / len(y)


def test_loaded_model_reproduces_recorded_evaluation(cremad_pkg, utd_pkg):
    c = ROOT / "data" / "cremad" / "cache"
    a = torch.from_numpy(np.load(c / "val_audio.npy")).float(); v = torch.from_numpy(np.load(c / "val_video.npy")).float()
    y = torch.from_numpy(np.load(c / "val_y.npy"))
    rec = cremad_pkg.evaluation["multimodal_joint_head"]["validation"]["accuracy"]
    assert round(_accuracy(cremad_pkg, a, v, y) * len(y)) == round(rec * len(y))
    u = ROOT / "data" / "utd_mhad" / "cache"
    a = torch.from_numpy(np.load(u / "test_depth.npy")).float(); v = torch.from_numpy(np.load(u / "test_inertial.npy")).float()
    y = torch.from_numpy(np.load(u / "test_y.npy"))
    rec = utd_pkg.evaluation["multimodal_joint_head"]["test_evaluation_only"]["accuracy"]
    assert round(_accuracy(utd_pkg, a, v, y) * len(y)) == round(rec * len(y))


def test_prediction_is_reproducible(cremad_predictor, cremad_val, tmp_path):
    from pathlib import Path
    r = cremad_val.iloc[5]
    kw = dict(audio=(Path(r.audio_path), "wav"), video=(Path(r.video_path), "flv"))
    p1 = cremad_predictor.predict(request_id="a", workdir=tmp_path, **kw).primary.probabilities
    p2 = cremad_predictor.predict(request_id="b", workdir=tmp_path, **kw).primary.probabilities
    assert [x.probability for x in p1] == [x.probability for x in p2]


def test_ood_fit_and_threshold_protocol(cremad_pkg, utd_pkg):
    for pkg in (cremad_pkg, utd_pkg):
        for name, o in pkg.evaluation["ood_detector"].items():
            assert o["fit_on"] == "train" and "validation" in o["threshold_from"]
            assert o["threshold"] == pytest.approx(pkg.detectors[name].threshold)
    meta = json.loads((ROOT / "checkpoints" / "cremad" / "baseline" / "metadata.json").read_text())
    assert meta["trained_on"] == "train split only"
