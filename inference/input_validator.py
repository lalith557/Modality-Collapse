"""File-content validation (magic bytes, not just names or MIME headers).

Every upload is sniffed from its first bytes before any decoder touches it. The detected
kind decides which decoder may run; a mismatch between the declared role (audio, video,
depth, inertial) and the content is rejected with an explicit error code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Hard limits (bytes). Uploads larger than these are rejected before decoding.
LIMITS = {"audio": 25 * 1024**2, "video": 100 * 1024**2, "depth": 64 * 1024**2, "inertial": 4 * 1024**2}

AUDIO_KINDS = {"wav", "flac", "ogg", "mp3", "mp4", "webm"}          # mp4/webm = m4a / browser recordings
VIDEO_KINDS = {"mp4", "webm", "avi", "flv", "mov"}
DEPTH_KINDS = {"mat5", "npy"}
INERTIAL_KINDS = {"mat5", "csv"}


class InputError(Exception):
    """A validation/decoding failure that should become a 4xx response."""

    def __init__(self, code: str, message: str, http_status: int = 422, field: str | None = None):
        super().__init__(message)
        self.code, self.message, self.http_status, self.field = code, message, http_status, field

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "field": self.field}


@dataclass
class Sniffed:
    kind: str
    detail: str


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def sniff(head: bytes) -> Sniffed | None:
    """Identify a file from its first bytes (>= 64 bytes recommended)."""
    if len(head) < 12:
        return None
    # A Git LFS pointer is a ~130-byte text stub left where the real file should be when a
    # repository (e.g. the CREMA-D GitHub repo) is cloned without `git lfs pull`.
    if head.startswith(LFS_POINTER_PREFIX):
        return Sniffed("lfs_pointer", "Git LFS pointer")
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return Sniffed("wav", "RIFF/WAVE")
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return Sniffed("avi", "RIFF/AVI")
    if head[:4] == b"fLaC":
        return Sniffed("flac", "FLAC")
    if head[:4] == b"OggS":
        return Sniffed("ogg", "Ogg")
    if head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return Sniffed("mp3", "MPEG audio")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        return Sniffed("mov" if brand == b"qt  " else "mp4", f"ISO-BMFF brand {brand!r}")
    if head[:4] == b"\x1aE\xdf\xa3":
        return Sniffed("webm", "Matroska/WebM")
    if head[:3] == b"FLV":
        return Sniffed("flv", "FLV")
    if head[:6] == b"\x93NUMPY":
        return Sniffed("npy", "NumPy .npy")
    if head[:10] == b"MATLAB 5.0":
        return Sniffed("mat5", "MATLAB v5 .mat")
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return Sniffed("mat73", "MATLAB v7.3 / HDF5")
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [l for l in text.replace("\r", "").split("\n") if l.strip()]
    if lines and all(sum(ch.isdigit() for ch in l) > 0 for l in lines[1:3] or lines[:1]):
        if any(sep in lines[0] for sep in (",", ";", "\t", " ")):
            return Sniffed("csv", "delimited text")
    return None


def check_file(path: Path, role: str) -> Sniffed:
    """Validate an uploaded file for a role: audio | video | depth | inertial."""
    size = path.stat().st_size
    if size == 0:
        raise InputError("empty_file", f"The {role} file is empty.", 400, role)
    if size > LIMITS[role]:
        raise InputError("file_too_large", f"The {role} file is {size/1e6:.1f} MB; the limit is "
                         f"{LIMITS[role]/1e6:.0f} MB.", 413, role)
    with open(path, "rb") as f:
        sn = sniff(f.read(64))
    allowed = {"audio": AUDIO_KINDS, "video": VIDEO_KINDS, "depth": DEPTH_KINDS, "inertial": INERTIAL_KINDS}[role]
    if sn is not None and sn.kind == "lfs_pointer":
        with open(path, "rb") as f:
            pointer = f.read(512).decode("utf-8", errors="replace")
        real = next((l.split()[1] for l in pointer.splitlines() if l.startswith("size ") and len(l.split()) == 2), None)
        real_txt = f" (the real file is {int(real):,} bytes)" if real and real.isdigit() else ""
        raise InputError("lfs_pointer", f"The {role} upload is a {size}-byte Git LFS placeholder, not the "
                         f"actual {role} file{real_txt}. The repository was downloaded without its "
                         "large files: run `git lfs pull` in it, or upload the copy from a full "
                         "download of the dataset.", 415, role)
    if sn is None:
        raise InputError("unrecognized_format", f"The {role} file's content is not a recognized "
                         f"format. Accepted: {', '.join(sorted(allowed))}.", 415, role)
    if sn.kind == "mat73":
        raise InputError("unsupported_format", "MATLAB v7.3 (HDF5) files are not supported; save "
                         "the array with MATLAB's '-v7' option or as .npy/.csv.", 415, role)
    if sn.kind not in allowed:
        raise InputError("wrong_content_type", f"The {role} upload contains {sn.detail} data, which "
                         f"is not accepted for {role}. Accepted: {', '.join(sorted(allowed))}.", 415, role)
    return sn
