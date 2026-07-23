from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse

from .. import job_store
from ..config import settings
from ..execution_log import read_recent_events, read_recent_text
from ..queue import enqueue_run
from ..ratelimit import limiter
from ..validation import RunConfig

router = APIRouter()
DEFAULT_LOG_LIMIT = min(100, settings.execution_log_max_read)


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
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return {
        "job_id": record["job_id"],
        "status": record["status"],
        "queue_position": record["queue_position"],
        "effective_config": record["effective_config"],
    }


@router.get("/execution-logs")
async def get_execution_logs(
    limit: int = Query(
        default=DEFAULT_LOG_LIMIT,
        ge=1,
        le=settings.execution_log_max_read,
    ),
) -> dict:
    if not settings.expose_execution_logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution logs are not exposed.",
        )
    return {
        "rotation": {
            "rotation_days": settings.execution_log_rotation_days,
            "max_bytes": settings.execution_log_max_bytes,
            "retention_files": settings.execution_log_retention_files,
        },
        "events": await read_recent_events(limit),
    }


@router.get("/execution-logs.txt", response_class=PlainTextResponse)
async def get_execution_logs_text(
    max_chars: int = Query(default=50_000, ge=1_000, le=200_000),
) -> PlainTextResponse:
    if not settings.expose_execution_logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution logs are not exposed.",
        )
    text = await read_recent_text(max_chars)
    return PlainTextResponse(text or "No execution logs yet.\n")


@router.get("/run/{job_id}")
async def get_run(job_id: str) -> dict:
    record = await job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return record
