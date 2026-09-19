"""Config loading and machine-readable run records (provenance)."""
from __future__ import annotations

import copy
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars to plain python for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return obj


class RunRecord:
    """Accumulates provenance + metrics for a single run and writes JSON."""

    def __init__(self, config: dict, extra: dict | None = None):
        self.data: dict = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_hash": git_hash(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "config": copy.deepcopy(config),
        }
        if extra:
            self.data.update(extra)

    def set(self, **kw) -> "RunRecord":
        self.data.update(kw)
        return self

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(self.data), f, indent=2)
        return path
