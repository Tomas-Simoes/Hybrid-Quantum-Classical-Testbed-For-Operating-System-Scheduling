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


def _refresh_queue_positions_locked() -> None:
    queued_jobs = sorted(
        (
            record
            for record in _jobs.values()
            if record.get("status") == "queued"
        ),
        key=lambda record: record.get("submitted_at", 0),
    )
    for position, record in enumerate(queued_jobs, start=1):
        record["queue_position"] = position


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
    _refresh_queue_positions_locked()


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(record)
    public["queue_capacity"] = settings.public_max_queue_size
    public["queue_running_count"] = sum(
        1 for job in _jobs.values() if job.get("status") == "running"
    )
    public.pop("request", None)
    public.pop("internal_error", None)
    return public


def _active_records_locked() -> list[dict[str, Any]]:
    return [
        record
        for record in _jobs.values()
        if record.get("status") not in TERMINAL_STATUSES
    ]


def _client_host(record: dict[str, Any]) -> str | None:
    request = record.get("request")
    if not isinstance(request, dict):
        return None
    host = request.get("client_host")
    return str(host) if host else None


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
        _refresh_queue_positions_locked()
        return deepcopy(record)


async def mark_running(job_id: str) -> dict[str, Any] | None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["queue_position"] = 0
            _jobs[job_id]["started_at"] = time.time()
            _jobs[job_id]["updated_at"] = _jobs[job_id]["started_at"]
            _refresh_queue_positions_locked()
            return deepcopy(_jobs[job_id])
        return None


async def mark_done(job_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["updated_at"] = time.time()
            _cleanup_locked(_jobs[job_id]["updated_at"])
            _refresh_queue_positions_locked()
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
            _refresh_queue_positions_locked()
            return deepcopy(_jobs[job_id])
        return None


async def delete_job(job_id: str) -> None:
    async with _lock:
        _jobs.pop(job_id, None)
        _refresh_queue_positions_locked()


async def get_job(job_id: str) -> dict[str, Any] | None:
    async with _lock:
        _cleanup_locked()
        _refresh_queue_positions_locked()
        record = _jobs.get(job_id)
        return _public_record(record) if record else None


async def active_counts(client_host: str | None = None) -> dict[str, int]:
    async with _lock:
        _cleanup_locked()
        active_records = _active_records_locked()
        queued_records = [
            record for record in active_records if record.get("status") == "queued"
        ]
        running_records = [
            record for record in active_records if record.get("status") == "running"
        ]
        client_records = [
            record for record in active_records if client_host and _client_host(record) == client_host
        ]
        return {
            "active": len(active_records),
            "queued": len(queued_records),
            "running": len(running_records),
            "client_active": len(client_records),
            "client_queued": sum(1 for record in client_records if record.get("status") == "queued"),
            "client_running": sum(1 for record in client_records if record.get("status") == "running"),
        }
