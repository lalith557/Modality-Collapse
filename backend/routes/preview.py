"""POST /preview/video: frames for files the browser cannot play. Uploads live only in a
per-request TemporaryDirectory; no model is run."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.routes.predict import _err
from backend.services.file_validation import save_upload
from inference.input_validator import InputError
from inference.video_preview import video_preview

router = APIRouter(prefix="/preview")
log = logging.getLogger("studio")


@router.post("/video")
async def preview_video(request: Request, video: UploadFile = File(..., description="video to preview")):
    rid = uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="studio_") as tmp:
        try:
            path, kind, size = await save_upload(video, "video", Path(tmp))
            log.info("rid=%s preview kind=%s size=%d", rid, kind, size)      # never content
            body = await request.app.state.service.run(video_preview, path=path, request_id=rid)
            return JSONResponse(content=body, headers={"X-Request-ID": rid})
        except InputError as e:
            return _err(rid, e.http_status, e.code, e.message, e.field)
        except asyncio.TimeoutError:
            return _err(rid, 504, "preview_timeout", "Decoding the preview exceeded its time budget.")
        except Exception:  # noqa: BLE001
            log.exception("rid=%s preview internal error", rid)
            return _err(rid, 500, "internal_error", "An internal error occurred while decoding the preview.")
