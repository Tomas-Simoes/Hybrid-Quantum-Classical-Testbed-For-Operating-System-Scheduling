from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from typing import Any, Literal

from .config import settings
from .validation import RunConfig

JobStatus = Literal["queued", "running", "done", "failed"]
TERMINAL_STATUSES = {"done", "failed"}
PUBLIC_FAILURE_MESSAGE = "Run failed. Reduce workload size or retry shortly."

_jobs: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()


def _cleanup_locked(now: float | None = None) -> None:
    now = now or time.time()
    ttl_cutoff = now - settings.public_job_ttl_seconds

    for job_id, record in list(_jobs.items()):
        if record.get("status") in TERMINAL_STATUSES and record.get("updated_at", 0) < ttl_cutoff:
            del _jobs[job_id]

    terminal_jobs = sorted(
        (
            (job_id, record)
            for job_id, record in _jobs.items()
            if record.get("status") in TERMINAL_STATUSES
        ),
        key=lambda item: item[1].get("updated_at", 0),
        reverse=True,
    )
    for job_id, _record in terminal_jobs[settings.public_max_completed_jobs :]:
        del _jobs[job_id]


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(record)
    public.pop("request", None)
    public.pop("internal_error", None)
    return public


async def create_job(
    job_id: str,
    config: RunConfig,
    queue_position: int,
    request_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = time.time()
    record = {
        "job_id": job_id,
        "status": "queued",
        "submitted_at": now,
        "updated_at": now,
        "queue_position": queue_position,
        "request": request_metadata or {},
        "effective_config": config.model_dump(),
        "result": None,
        "error": None,
    }
    async with _lock:
        _cleanup_locked(now)
        _jobs[job_id] = record
        return deepcopy(record)


async def mark_running(job_id: str) -> dict[str, Any] | None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["queue_position"] = 0
            _jobs[job_id]["started_at"] = time.time()
            _jobs[job_id]["updated_at"] = _jobs[job_id]["started_at"]
            return deepcopy(_jobs[job_id])
        return None


async def mark_done(job_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["updated_at"] = time.time()
            _cleanup_locked(_jobs[job_id]["updated_at"])
            return deepcopy(_jobs[job_id])
        return None


async def mark_failed(job_id: str, error: BaseException) -> dict[str, Any] | None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = {
                "type": "JobFailed",
                "message": PUBLIC_FAILURE_MESSAGE,
            }
            _jobs[job_id]["internal_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            _jobs[job_id]["updated_at"] = time.time()
            _cleanup_locked(_jobs[job_id]["updated_at"])
            return deepcopy(_jobs[job_id])
        return None


async def get_job(job_id: str) -> dict[str, Any] | None:
    async with _lock:
        _cleanup_locked()
        record = _jobs.get(job_id)
        return _public_record(record) if record else None
