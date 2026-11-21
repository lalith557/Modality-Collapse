"""Prediction Studio API (research prototype).

Run from the repository root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from backend.routes import health, predict, preview  # noqa: E402
from backend.services.inference_service import InferenceService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = InferenceService()
    svc.load()                       # loads + hash-verifies checkpoints, warms up the models
    app.state.service = svc
    yield


app = FastAPI(title="Prediction Studio API", version="1.0.0", lifespan=lifespan,
              description="Research-prototype inference for the 'Feature Scale as a Hidden Per-Modality "
                          "Learning Rate' project. Not a production emotion- or action-recognition system.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"], expose_headers=["X-Request-ID"])


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    missing = [".".join(str(x) for x in e["loc"][1:]) for e in exc.errors()]
    return JSONResponse(status_code=400, content={"request_id": None, "error": {
        "code": "invalid_request", "message": f"Missing or invalid fields: {', '.join(missing) or 'request body'}.",
        "field": missing[0] if missing else None}})


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(preview.router)
