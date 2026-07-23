from __future__ import annotations

import asyncio
import json
import logging
import math
import traceback
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)

_log_lock = asyncio.Lock()
SECONDS_PER_DAY = 86_400


def _utc_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
            "traceback": "".join(
                traceback.format_exception(type(value), value, value.__traceback__)
            ),
        }
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _archive_files(path: Path) -> list[Path]:
    if not path.parent.exists():
        return []
    return sorted(
        (
            archive
            for archive in path.parent.glob(f"{path.stem}-*{path.suffix}")
            if archive.is_file()
        ),
        key=lambda archive: archive.name,
    )


def _unique_archive_path(path: Path, rotated_at: datetime) -> Path:
    stamp = rotated_at.strftime("%Y%m%d_%H%M%S_%f")
    counter = 0
    while True:
        candidate = path.with_name(
            f"{path.stem}-{stamp}-{counter:03d}{path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def _active_log_start_unix(path: Path) -> float | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    break
                timestamp = event.get("timestamp_unix")
                if isinstance(timestamp, (int, float)):
                    return float(timestamp)
                break
        return path.stat().st_mtime
    except FileNotFoundError:
        return None
    except OSError:
        logger.exception("Failed to inspect backend execution log for rotation.")
        return None


def _should_rotate(path: Path, line: str, now_unix: float) -> bool:
    try:
        current_size = path.stat().st_size
    except FileNotFoundError:
        return False
    except OSError:
        logger.exception("Failed to stat backend execution log for rotation.")
        return False

    if current_size <= 0:
        return False

    next_size = current_size + len(line.encode("utf-8")) + 1
    if next_size > settings.execution_log_max_bytes:
        return True

    start_unix = _active_log_start_unix(path)
    if start_unix is None:
        return False

    rotation_seconds = settings.execution_log_rotation_days * SECONDS_PER_DAY
    return now_unix - start_unix >= rotation_seconds


def _prune_archives(path: Path) -> None:
    archives = _archive_files(path)
    excess_count = len(archives) - settings.execution_log_retention_files
    if excess_count <= 0:
        return

    for archive in archives[:excess_count]:
        try:
            archive.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Failed to prune backend execution log archive.")


def _rotate_if_needed(path: Path, line: str, now: datetime, now_unix: float) -> None:
    if not _should_rotate(path, line, now_unix):
        return

    try:
        archive_path = _unique_archive_path(path, now)
        path.rename(archive_path)
        _prune_archives(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.exception("Failed to rotate backend execution log.")


def _write_line(path: Path, line: str, now: datetime, now_unix: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed(path, line, now, now_unix)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


async def record_event(event: str, **payload: Any) -> None:
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    timestamp_unix = now.timestamp()
    entry = {
        "event": event,
        "timestamp": timestamp,
        "timestamp_unix": timestamp_unix,
        **payload,
    }
    line = json.dumps(_jsonable(entry), ensure_ascii=True, sort_keys=True)

    try:
        async with _log_lock:
            await asyncio.to_thread(
                _write_line,
                settings.execution_log_path,
                line,
                now,
                timestamp_unix,
            )
    except Exception:
        logger.exception("Failed to write backend execution log event.")


async def log_job_queued(record: dict[str, Any]) -> None:
    await record_event(
        "execution.queued",
        job_id=record.get("job_id"),
        status=record.get("status"),
        submitted_at=record.get("submitted_at"),
        submitted_at_iso=_utc_iso(record.get("submitted_at")),
        queue_position=record.get("queue_position"),
        request=record.get("request"),
        effective_config=record.get("effective_config"),
    )


async def log_job_started(record: dict[str, Any] | None) -> None:
    record = record or {}
    await record_event(
        "execution.started",
        job_id=record.get("job_id"),
        status=record.get("status", "running"),
        submitted_at=record.get("submitted_at"),
        submitted_at_iso=_utc_iso(record.get("submitted_at")),
        started_at=record.get("started_at"),
        started_at_iso=_utc_iso(record.get("started_at")),
        request=record.get("request"),
        effective_config=record.get("effective_config"),
    )


async def log_job_completed(record: dict[str, Any] | None) -> None:
    record = record or {}
    result = record.get("result") or {}
    await record_event(
        "execution.completed",
        job_id=record.get("job_id"),
        status=record.get("status", "done"),
        submitted_at=record.get("submitted_at"),
        submitted_at_iso=_utc_iso(record.get("submitted_at")),
        started_at=record.get("started_at"),
        started_at_iso=_utc_iso(record.get("started_at")),
        completed_at=record.get("updated_at"),
        completed_at_iso=_utc_iso(record.get("updated_at")),
        duration_ms=result.get("duration_ms"),
        request=record.get("request"),
        effective_config=record.get("effective_config"),
        result=result,
    )


async def log_job_failed(record: dict[str, Any] | None, error: BaseException) -> None:
    record = record or {}
    await record_event(
        "execution.failed",
        job_id=record.get("job_id"),
        status=record.get("status", "failed"),
        submitted_at=record.get("submitted_at"),
        submitted_at_iso=_utc_iso(record.get("submitted_at")),
        started_at=record.get("started_at"),
        started_at_iso=_utc_iso(record.get("started_at")),
        failed_at=record.get("updated_at"),
        failed_at_iso=_utc_iso(record.get("updated_at")),
        request=record.get("request"),
        effective_config=record.get("effective_config"),
        error=error,
    )


def _read_recent(files: list[Path], limit: int) -> list[dict[str, Any]]:
    events: deque[dict[str, Any]] = deque(maxlen=limit)
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError:
                    events.append(
                        {
                            "event": "execution_log.invalid_line",
                            "line_number": line_number,
                            "raw": stripped,
                            "source_file": path.name,
                        }
                    )
    return list(events)


def log_files_for_read(path: Path | None = None) -> list[Path]:
    active_path = path or settings.execution_log_path
    files = _archive_files(active_path)
    if active_path.exists():
        files.append(active_path)
    return files


async def read_recent_events(limit: int | None = None) -> list[dict[str, Any]]:
    requested_limit = limit or settings.execution_log_max_read
    bounded_limit = max(1, min(int(requested_limit), settings.execution_log_max_read))
    async with _log_lock:
        return await asyncio.to_thread(
            _read_recent,
            log_files_for_read(settings.execution_log_path),
            bounded_limit,
        )
