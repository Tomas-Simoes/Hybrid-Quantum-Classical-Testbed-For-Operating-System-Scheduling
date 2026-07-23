from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from . import job_store
from .config import settings
from .execution_log import (
    log_job_completed,
    log_job_failed,
    log_job_queued,
    log_job_started,
)
from .pipeline_bridge import run_pipeline
from .validation import RunConfig


@dataclass(frozen=True)
class QueuedJob:
    job_id: str
    config: RunConfig


_queue: asyncio.Queue[QueuedJob] | None = None
_worker_task: asyncio.Task[None] | None = None


def _job_queue() -> asyncio.Queue[QueuedJob]:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue(maxsize=settings.public_max_queue_size)
    return _queue


async def enqueue_run(
    config: RunConfig,
    request_metadata: dict[str, Any] | None = None,
) -> dict:
    queue = _job_queue()
    if queue.full():
        raise RuntimeError("Job queue is full. Please retry shortly.")

    job_id = str(uuid.uuid4())
    record = await job_store.create_job(
        job_id,
        config,
        queue.qsize() + 1,
        request_metadata,
    )
    await queue.put(QueuedJob(job_id=job_id, config=config))
    await log_job_queued(record)
    return record


async def start_worker() -> None:
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker(), name="adapter-scheduler-worker")


async def stop_worker() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    _worker_task = None


async def _worker() -> None:
    queue = _job_queue()
    while True:
        job = await queue.get()
        try:
            running_record = await job_store.mark_running(job.job_id)
            await log_job_started(running_record)
            result = await asyncio.to_thread(run_pipeline, job.config)
            done_record = await job_store.mark_done(job.job_id, result)
            await log_job_completed(done_record)
        except Exception as exc:
            failed_record = await job_store.mark_failed(job.job_id, exc)
            await log_job_failed(failed_record, exc)
        finally:
            queue.task_done()
