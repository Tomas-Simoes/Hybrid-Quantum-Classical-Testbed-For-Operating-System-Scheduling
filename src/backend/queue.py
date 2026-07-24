from __future__ import annotations

import asyncio
import json
import multiprocessing
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
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


class QueueAdmissionError(RuntimeError):
    status_code = 503
    public_message = "Job queue is full. Please retry shortly."


class QueueFullError(QueueAdmissionError):
    pass


class ClientJobLimitError(QueueAdmissionError):
    status_code = 429
    public_message = "Too many active jobs from this client. Wait for one to finish."


class JobTimeoutError(TimeoutError):
    pass


class PipelineChildError(RuntimeError):
    pass


_queue: asyncio.Queue[QueuedJob] | None = None
_worker_task: asyncio.Task[None] | None = None
_admission_lock = asyncio.Lock()


def _job_queue() -> asyncio.Queue[QueuedJob]:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue(maxsize=settings.public_max_queue_size)
    return _queue


async def enqueue_run(
    config: RunConfig,
    request_metadata: dict[str, Any] | None = None,
) -> dict:
    metadata = request_metadata or {}
    client_host = metadata.get("client_host")

    async with _admission_lock:
        queue = _job_queue()
        counts = await job_store.active_counts(str(client_host) if client_host else None)
        if counts["active"] >= settings.public_max_active_jobs or queue.full():
            raise QueueFullError(QueueFullError.public_message)
        if counts["client_active"] >= settings.public_max_jobs_per_ip:
            raise ClientJobLimitError(ClientJobLimitError.public_message)

        job_id = str(uuid.uuid4())
        record = await job_store.create_job(
            job_id,
            config,
            queue.qsize() + 1,
            metadata,
        )
        try:
            queue.put_nowait(QueuedJob(job_id=job_id, config=config))
        except asyncio.QueueFull as exc:
            await job_store.delete_job(job_id)
            raise QueueFullError(QueueFullError.public_message) from exc

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
            result = await asyncio.to_thread(
                _run_pipeline_process,
                job.config,
                settings.public_job_timeout_seconds,
            )
            done_record = await job_store.mark_done(job.job_id, result)
            await log_job_completed(done_record)
        except Exception as exc:
            failed_record = await job_store.mark_failed(job.job_id, exc)
            await log_job_failed(failed_record, exc)
        finally:
            queue.task_done()


def _pipeline_child(config_data: dict[str, Any], result_path: str) -> None:
    try:
        result = run_pipeline(RunConfig.model_validate(config_data))
        payload = {"status": "ok", "result": result}
    except BaseException as exc:
        payload = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        }

    Path(result_path).write_text(
        json.dumps(payload, ensure_ascii=True),
        encoding="utf-8",
    )


def _terminate_process(process: multiprocessing.Process) -> None:
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _process_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("forkserver")
    except ValueError:
        return multiprocessing.get_context("spawn")


def _load_pipeline_payload(result_path: Path) -> dict[str, Any]:
    try:
        raw = result_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise PipelineChildError("Pipeline process did not write a result.") from exc
    if not raw:
        raise PipelineChildError("Pipeline process wrote an empty result.")
    return json.loads(raw)


def _run_pipeline_process(config: RunConfig, timeout_seconds: int) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        prefix="hybrid-scheduler-job-",
        suffix=".json",
        delete=False,
    ) as handle:
        result_path = Path(handle.name)

    process = _process_context().Process(
        target=_pipeline_child,
        args=(config.model_dump(), str(result_path)),
        daemon=True,
    )
    try:
        process.start()
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            _terminate_process(process)
            raise JobTimeoutError(
                f"Pipeline job exceeded {timeout_seconds} second timeout."
            )

        try:
            payload = _load_pipeline_payload(result_path)
        except json.JSONDecodeError as exc:
            raise PipelineChildError("Pipeline process wrote invalid JSON.") from exc

        if payload.get("status") == "ok":
            return payload["result"]

        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        error_type = error.get("type", "PipelineError")
        error_message = error.get("message", "Pipeline process failed.")
        raise PipelineChildError(f"{error_type}: {error_message}")
    finally:
        if process.is_alive():
            _terminate_process(process)
        try:
            result_path.unlink()
        except FileNotFoundError:
            pass
