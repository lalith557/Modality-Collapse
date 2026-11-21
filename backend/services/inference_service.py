"""Owns the loaded checkpoint packages and runs predictions safely.

* Packages are loaded and hash-verified once at startup; a package that fails verification
  is reported by /health and its endpoints return 503 (the service never falls back to an
  unverified model).
* Each predictor serializes its own forward passes (a lock inside the predictor); a global
  semaphore bounds concurrent CPU-heavy decoding, and every prediction has a time budget.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import torch

from inference.cremad_predictor import CremadPredictor
from inference.model_loader import load_package
from inference.utd_predictor import UtdPredictor

log = logging.getLogger("studio")
PREDICT_TIMEOUT_S = 120.0
MAX_CONCURRENT = 2


class InferenceService:
    def __init__(self):
        self.started = time.time()
        self.cremad: CremadPredictor | None = None
        self.utd: UtdPredictor | None = None
        self.errors: dict[str, str] = {}
        self.sem = asyncio.Semaphore(MAX_CONCURRENT)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        for name in ("cremad", "utd_mhad"):
            try:
                pkg = load_package(name, self.device)
                if name == "cremad":
                    self.cremad = CremadPredictor(pkg)
                else:
                    self.utd = UtdPredictor(pkg)
                self._warmup(pkg)
                log.info("loaded %s (%d files hash-verified)", pkg.checkpoint_id, len(pkg.files_verified))
            except Exception as e:  # noqa: BLE001
                self.errors[name] = f"{type(e).__name__}: {e}"
                log.error("failed to load %s: %s", name, self.errors[name])

    @staticmethod
    def _warmup(pkg) -> None:
        m = pkg.model
        with torch.inference_mode():
            if pkg.dataset == "cremad":
                m(torch.zeros(1, 1, 64, 188, device=pkg.device), torch.zeros(1, 8, 1, 64, 64, device=pkg.device))
            else:
                m(torch.zeros(1, 16, 1, 32, 32, device=pkg.device), torch.zeros(1, 6, 128, device=pkg.device))

    def models_status(self) -> dict:
        out = {}
        for name, pred in (("cremad", self.cremad), ("utd_mhad", self.utd)):
            out[name] = ({"loaded": True, "checkpoint_id": pred.pkg.checkpoint_id,
                          "files_verified": pred.pkg.files_verified}
                         if pred else {"loaded": False, "error": self.errors.get(name, "not loaded")})
        return out

    async def run(self, fn, /, **kwargs):
        async with self.sem:
            return await asyncio.wait_for(asyncio.to_thread(fn, **kwargs), timeout=PREDICT_TIMEOUT_S)
