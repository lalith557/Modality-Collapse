"""Size-limited, time-limited streaming of uploads to a per-request temp dir + content sniffing.

Uploads are never written anywhere except the request's TemporaryDirectory, which the route
deletes when the request ends (success or failure). Raw contents are never logged.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import UploadFile

from inference.input_validator import LIMITS, InputError, check_file

CHUNK = 1 << 16
READ_TIMEOUT_S = 30.0


async def save_upload(upload: UploadFile, role: str, workdir: Path) -> tuple[Path, str, int]:
    """Stream an upload to disk, enforcing the size limit while reading; then sniff its content."""
    dest = workdir / f"upload_{role}"
    limit = LIMITS[role]
    size = 0
    with open(dest, "wb") as f:
        while True:
            try:
                chunk = await asyncio.wait_for(upload.read(CHUNK), timeout=READ_TIMEOUT_S)
            except asyncio.TimeoutError:
                raise InputError("upload_timeout", f"Reading the {role} upload timed out.", 408, role)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise InputError("file_too_large", f"The {role} file exceeds the {limit/1e6:.0f} MB limit.", 413, role)
            f.write(chunk)
    sn = check_file(dest, role)
    return dest, sn.kind, size
