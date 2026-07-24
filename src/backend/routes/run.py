from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from .. import job_store
from ..config import settings
from ..queue import QueueAdmissionError, enqueue_run
from ..ratelimit import limiter
from ..validation import RunConfig

router = APIRouter()


def _request_metadata(request: Request) -> dict:
    return {
        "method": request.method,
        "path": request.url.path,
        "query": request.url.query,
        "client_host": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(settings.run_rate_limit)
async def create_run(request: Request, config: RunConfig) -> dict:
    try:
        record = await enqueue_run(config, _request_metadata(request))
    except QueueAdmissionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    public_record = await job_store.get_job(record["job_id"]) or record
    return {
        "job_id": public_record["job_id"],
        "status": public_record["status"],
        "queue_position": public_record["queue_position"],
        "queue_capacity": public_record.get("queue_capacity", settings.public_max_queue_size),
        "queue_running_count": public_record.get("queue_running_count", 0),
        "effective_config": public_record["effective_config"],
    }


@router.get("/run/{job_id}")
@limiter.limit(settings.poll_rate_limit)
async def get_run(request: Request, job_id: str) -> dict:
    record = await job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return record
