"""Device selection. The project is GPU-first (user requirement)."""
from __future__ import annotations

import torch


def get_device(cfg_device: dict | None = None) -> torch.device:
    """Return the compute device.

    Defaults to CUDA. If CUDA is unavailable and ``allow_cpu`` is not set, raise
    loudly rather than silently falling back to CPU -- experiments in this project
    are required to run on the GPU.
    """
    cfg_device = cfg_device or {}
    want = str(cfg_device.get("type", "cuda")).lower()
    allow_cpu = bool(cfg_device.get("allow_cpu", False))

    if want == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Set device.allow_cpu: true to override, or fix the CUDA/torch install."
        )
    return torch.device(want)


def device_report(device: torch.device) -> dict:
    """Machine-readable snapshot of the active device."""
    info: dict = {"device": str(device)}
    if device.type == "cuda":
        idx = device.index or 0
        info.update(
            name=torch.cuda.get_device_name(idx),
            capability=list(torch.cuda.get_device_capability(idx)),
            total_mem_mb=round(torch.cuda.get_device_properties(idx).total_memory / 1e6, 1),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    else:
        info.update(torch_version=torch.__version__, cuda_version=torch.version.cuda)
    return info
