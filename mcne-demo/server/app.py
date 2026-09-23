from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trainer import run_training


API_KEY = os.getenv("MCNE_API_KEY", "").strip()
DEVICE_MODE = os.getenv("DEVICE_MODE", "auto")
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data/Cora")).resolve()
CHECKPOINT_ROOT = Path(os.getenv("CHECKPOINT_ROOT", "./checkpoints")).resolve()
ALLOWED_ORIGINS = [value.strip() for value in os.getenv(
    "ALLOWED_ORIGINS", "https://wwwsaidthat.github.io,http://localhost:8000"
).split(",") if value.strip()]

app = FastAPI(title="MCNE GPU Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class TrainRequest(BaseModel):
    epochs: int = Field(default=100, ge=10, le=1000)
    seed: int = Field(default=42, ge=0, le=999999)
    dimensions: list[int] = Field(default=[32, 64, 128, 256, 384, 512, 768])
    max_dimension: int = Field(default=768, ge=32, le=2048)
    batch_size: int = Field(default=512, ge=64, le=1024)


jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcne-gpu")


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


def patch_job(job_id: str, patch: dict) -> None:
    with jobs_lock:
        jobs[job_id].update(patch)
        jobs[job_id]["updated_at"] = time.time()


def execute_job(job_id: str, config: dict) -> None:
    patch_job(job_id, {"status": "running", "started_at": time.time()})

    def update(progress: dict) -> None:
        patch_job(job_id, progress)

    def should_cancel() -> bool:
        with jobs_lock:
            return bool(jobs[job_id].get("cancel_requested"))

    try:
        output = run_training(
            config,
            update,
            should_cancel,
            data_root=DATA_ROOT,
            checkpoint_root=CHECKPOINT_ROOT,
            device_mode=DEVICE_MODE,
        )
        patch_job(job_id, {"status": "completed", **output})
    except Exception as exc:
        status = "cancelled" if "cancelled" in str(exc).lower() else "failed"
        patch_job(job_id, {"status": status, "error": str(exc)})


@app.get("/api/health")
def health() -> dict:
    import torch
    return {
        "ok": True,
        "device_mode": DEVICE_MODE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "queue_busy": any(job.get("status") in {"queued", "running"} for job in jobs.values()),
    }


@app.post("/api/train", dependencies=[Depends(require_api_key)])
def start_training(request: TrainRequest) -> dict:
    with jobs_lock:
        if any(job.get("status") in {"queued", "running"} for job in jobs.values()):
            raise HTTPException(status_code=409, detail="the GPU worker is busy")
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "epoch": 0,
            "total_epochs": request.epochs,
            "loss": None,
            "elapsed_seconds": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "cancel_requested": False,
        }
    executor.submit(execute_job, job_id, request.model_dump())
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/train/{job_id}", dependencies=[Depends(require_api_key)])
def training_status(job_id: str) -> dict:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(jobs[job_id])


@app.delete("/api/train/{job_id}", dependencies=[Depends(require_api_key)])
def cancel_training(job_id: str) -> dict:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="job not found")
        jobs[job_id]["cancel_requested"] = True
    return {"job_id": job_id, "status": "cancellation requested"}
