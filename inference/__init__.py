"""Prediction Studio inference engine (research prototype).

Reuses the research project's model classes and *exact* preprocessing functions.
See docs/PREDICTION_STUDIO.md for the contract this package implements.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints"
