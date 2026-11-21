"""The website's generated data must match the model code, the checkpoints and the frozen results.

* architecture.generated.json is regenerated from the PyTorch modules (strict checkpoint load)
  and must be byte-identical to the committed file.
* evidence.verified.json is re-read from the frozen result artifacts and must be identical.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _check(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--check"],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_architecture_json_matches_model_code_and_checkpoints():
    r = _check("export_architecture.py")
    assert r.returncode == 0, r.stdout + r.stderr


def test_evidence_json_matches_frozen_artifacts():
    r = _check("export_site_evidence.py")
    assert r.returncode == 0, r.stdout + r.stderr


def test_architecture_totals_equal_checkpoint_metadata():
    arch = json.loads((ROOT / "website/src/data/architecture.generated.json").read_text(encoding="utf-8"))
    for ds, d in arch["datasets"].items():
        meta = json.loads((ROOT / "checkpoints" / ds / "baseline" / "metadata.json").read_text(encoding="utf-8"))
        assert d["params"]["total"] == meta["parameters"]
        assert sum(n["params"] for n in d["nodes"]) == meta["parameters"]
        assert d["checkpoint"]["sha256"] == meta["checkpoint_hash"]["model.pth"]


def test_lab_trajectories_match_frozen_logs():
    r = _check("export_lab_trajectories.py")
    assert r.returncode == 0, r.stdout + r.stderr
