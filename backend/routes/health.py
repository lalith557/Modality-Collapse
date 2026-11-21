"""GET /health and GET /models."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from backend.schemas.request import ACCEPTED, ENDPOINTS
from backend.schemas.response import HealthResponse
from inference.common import limitations

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    svc = request.app.state.service
    models = svc.models_status()
    ok = all(m["loaded"] for m in models.values())
    return HealthResponse(status="ok" if ok else "degraded", device=str(svc.device), models=models,
                          uptime_s=time.time() - svc.started)


@router.get("/models")
def models(request: Request):
    svc = request.app.state.service
    out = {"research_prototype": True, "accepted_inputs": ACCEPTED, "endpoints": ENDPOINTS, "models": {}}
    for name, pred in (("cremad", svc.cremad), ("utd_mhad", svc.utd)):
        if pred is None:
            out["models"][name] = {"loaded": False, "error": svc.errors.get(name)}
            continue
        p = pred.pkg
        q = p.quality_reference or {}
        out["models"][name] = {
            "loaded": True, "checkpoint_id": p.checkpoint_id, "metadata": p.metadata, "labels": p.labels,
            "preprocessing": p.preprocessing, "evaluation": p.evaluation,
            "quality_thresholds": {k: v for k, v in q.items() if k.endswith("_thresholds")},
            "quality_false_flag_rates": q.get("false_flag_rates"),
            "modalities": list(p.modalities), "limitations": limitations(p)}
    return out
