"""Shared fixtures for the Prediction Studio test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTERNAL = ROOT / "samples" / "external"


@pytest.fixture(scope="session")
def cremad_pkg():
    from inference.model_loader import load_package
    return load_package("cremad")


@pytest.fixture(scope="session")
def utd_pkg():
    from inference.model_loader import load_package
    return load_package("utd_mhad")


@pytest.fixture(scope="session")
def cremad_predictor(cremad_pkg):
    from inference.cremad_predictor import CremadPredictor
    return CremadPredictor(cremad_pkg)


@pytest.fixture(scope="session")
def utd_predictor(utd_pkg):
    from inference.utd_predictor import UtdPredictor
    return UtdPredictor(utd_pkg)


@pytest.fixture(scope="session")
def external() -> Path:
    """External (non-dataset) samples; generated on demand by scripts/make_external_samples.py."""
    if not all((EXTERNAL / n).exists() for n in ("face_with_speech.mp4", "face_zoom.flv")):
        from scripts.make_external_samples import main
        main(EXTERNAL)
    return EXTERNAL


@pytest.fixture(scope="session")
def cremad_val() -> pd.DataFrame:
    man = pd.read_csv(ROOT / "data" / "cremad" / "cremad_manifest.csv")
    ok = man[(man.audio_status == "present") & (man.video_status == "present")]
    return ok[ok.split == "val"].reset_index(drop=True)
