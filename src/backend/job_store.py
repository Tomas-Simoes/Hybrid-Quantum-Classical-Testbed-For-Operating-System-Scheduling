from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from typing import Any, Literal

from .validation import RunConfig

JobStatus = Literal["queued", "running", "done", "failed"]

_jobs: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()


async def create_job(job_id: str, config: RunConfig, queue_position: int) -> dict[str, Any]:
    now = time.time()
    record = {
        "job_id": job_id,
        "status": "queued",
        "submitted_at": now,
        "updated_at": now,
        "queue_position": queue_position,
        "effective_config": config.model_dump(),
        "result": None,
        "error": None,
    }
    async with _lock:
        _jobs[job_id] = record
        return deepcopy(record)


async def mark_running(job_id: str) -> None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["queue_position"] = 0
            _jobs[job_id]["started_at"] = time.time()
            _jobs[job_id]["updated_at"] = _jobs[job_id]["started_at"]


async def mark_done(job_id: str, result: dict[str, Any]) -> None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["updated_at"] = time.time()


async def mark_failed(job_id: str, error: BaseException) -> None:
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            _jobs[job_id]["updated_at"] = time.time()


async def get_job(job_id: str) -> dict[str, Any] | None:
    async with _lock:
        record = _jobs.get(job_id)
        return deepcopy(record) if record else None
