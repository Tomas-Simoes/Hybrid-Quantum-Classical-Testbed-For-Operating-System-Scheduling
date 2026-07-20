from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from .. import job_store
from ..config import settings
from ..queue import enqueue_run
from ..ratelimit import limiter
from ..validation import RunConfig

router = APIRouter()


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(settings.run_rate_limit)
async def create_run(request: Request, config: RunConfig) -> dict:
    try:
        record = await enqueue_run(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return {
        "job_id": record["job_id"],
        "status": record["status"],
        "queue_position": record["queue_position"],
        "effective_config": record["effective_config"],
    }


@router.get("/run/{job_id}")
async def get_run(job_id: str) -> dict:
    record = await job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return record
