"""Prediction endpoints. Uploads live only in a per-request TemporaryDirectory."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.services.file_validation import save_upload
from inference.input_validator import InputError
from inference.prediction_schema import PredictionResponse

router = APIRouter(prefix="/predict")
log = logging.getLogger("studio")


def _err(rid: str, status: int, code: str, message: str, field: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, headers={"X-Request-ID": rid},
                        content={"request_id": rid, "error": {"code": code, "message": message, "field": field}})


async def _handle(request: Request, dataset: str, uploads: dict[str, UploadFile | None], **opts):
    rid = uuid.uuid4().hex[:12]
    svc = request.app.state.service
    predictor = svc.cremad if dataset == "cremad" else svc.utd
    if predictor is None:
        return _err(rid, 503, "model_unavailable", f"The {dataset} checkpoint is not loaded: "
                    f"{svc.errors.get(dataset, 'unknown error')}")
    with tempfile.TemporaryDirectory(prefix="studio_") as tmp:
        workdir = Path(tmp)
        try:
            files, sizes = {}, {}
            for role, up in uploads.items():
                if up is None or not up.filename:
                    continue
                path, kind, size = await save_upload(up, role, workdir)
                files[role], sizes[role] = (path, kind), (kind, size)
            log.info("rid=%s dataset=%s inputs=%s", rid, dataset, sizes)      # sizes/kinds only, never content
            if dataset == "cremad":
                result: PredictionResponse = await svc.run(predictor.predict, request_id=rid, workdir=workdir,
                                                           audio=files.get("audio"), video=files.get("video"), **opts)
            else:
                result = await svc.run(predictor.predict, request_id=rid, workdir=workdir,
                                       depth=files.get("depth"), inertial=files.get("inertial"))
            log.info("rid=%s status=%s mode=%s total_ms=%.0f", rid, result.status, result.mode, result.total_ms)
            return JSONResponse(content=result.model_dump(), headers={"X-Request-ID": rid})
        except InputError as e:
            log.info("rid=%s rejected code=%s", rid, e.code)
            return _err(rid, e.http_status, e.code, e.message, e.field)
        except asyncio.TimeoutError:
            return _err(rid, 504, "inference_timeout", "Inference exceeded its time budget.")
        except Exception:  # noqa: BLE001
            log.exception("rid=%s internal error", rid)
            return _err(rid, 500, "internal_error", "An internal error occurred while processing the request.")


@router.post("/cremad/audio", response_model=PredictionResponse)
async def cremad_audio(request: Request, audio: UploadFile = File(..., description="speech recording")):
    return await _handle(request, "cremad", {"audio": audio})


@router.post("/cremad/video", response_model=PredictionResponse)
async def cremad_video(request: Request, video: UploadFile = File(..., description="frontal speaking-face video")):
    return await _handle(request, "cremad", {"video": video})


@router.post("/cremad/multimodal", response_model=PredictionResponse)
async def cremad_multimodal(request: Request, video: UploadFile = File(...),
                            audio: UploadFile | None = File(None),
                            use_video_audio: bool = Form(True)):
    if audio is None or not audio.filename:
        if not use_video_audio:
            return _err(uuid.uuid4().hex[:12], 400, "missing_modality",
                        "Multimodal inference needs audio: upload an audio file or set use_video_audio=true.", "audio")
        return await _handle(request, "cremad", {"video": video}, audio_from_video=True)
    return await _handle(request, "cremad", {"audio": audio, "video": video})


@router.post("/utd_mhad/multimodal", response_model=PredictionResponse)
async def utd_multimodal(request: Request, depth: UploadFile = File(...), inertial: UploadFile = File(...)):
    return await _handle(request, "utd_mhad", {"depth": depth, "inertial": inertial})
