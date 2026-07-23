from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from src.backend.config import settings
from src.backend.execution_log import (
    log_job_completed,
    log_job_failed,
    log_job_queued,
    record_event,
    read_recent_events,
)


class BackendExecutionLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_log_path = settings.execution_log_path
        self._previous_json_log_path = settings.execution_json_log_path
        self._previous_max_read = settings.execution_log_max_read
        self._previous_rotation_days = settings.execution_log_rotation_days
        self._previous_max_bytes = settings.execution_log_max_bytes
        self._previous_retention_files = settings.execution_log_retention_files
        self._tmpdir = tempfile.TemporaryDirectory()
        object.__setattr__(
            settings,
            "execution_log_path",
            Path(self._tmpdir.name) / "executions.log",
        )
        object.__setattr__(
            settings,
            "execution_json_log_path",
            Path(self._tmpdir.name) / "executions.jsonl",
        )
        object.__setattr__(settings, "execution_log_max_read", 50)
        object.__setattr__(settings, "execution_log_rotation_days", 14)
        object.__setattr__(settings, "execution_log_max_bytes", 10_000_000)
        object.__setattr__(settings, "execution_log_retention_files", 6)

    def tearDown(self) -> None:
        object.__setattr__(settings, "execution_log_path", self._previous_log_path)
        object.__setattr__(
            settings,
            "execution_json_log_path",
            self._previous_json_log_path,
        )
        object.__setattr__(settings, "execution_log_max_read", self._previous_max_read)
        object.__setattr__(
            settings,
            "execution_log_rotation_days",
            self._previous_rotation_days,
        )
        object.__setattr__(settings, "execution_log_max_bytes", self._previous_max_bytes)
        object.__setattr__(
            settings,
            "execution_log_retention_files",
            self._previous_retention_files,
        )
        self._tmpdir.cleanup()

    def test_execution_lifecycle_is_available_as_structured_events(self) -> None:
        async def scenario() -> list[dict]:
            queued_record = {
                "job_id": "job-1",
                "status": "queued",
                "submitted_at": 123.0,
                "queue_position": 1,
                "request": {
                    "method": "POST",
                    "path": "/api/run",
                    "client_host": "127.0.0.1",
                },
                "effective_config": {"num_processes": 2, "weights": [0.4, 0.6]},
            }
            completed_record = {
                **queued_record,
                "status": "done",
                "started_at": 124.0,
                "updated_at": 125.0,
                "result": {
                    "duration_ms": 1000.0,
                    "result": {"assignments": {"1000": 0}},
                },
            }

            await log_job_queued(queued_record)
            await log_job_completed(completed_record)
            return await read_recent_events()

        events = asyncio.run(scenario())

        self.assertEqual(
            [event["event"] for event in events],
            ["execution.queued", "execution.completed"],
        )
        self.assertEqual(events[0]["job_id"], "job-1")
        self.assertEqual(events[0]["request"]["path"], "/api/run")
        self.assertEqual(events[0]["effective_config"]["weights"], [0.4, 0.6])
        self.assertEqual(events[1]["duration_ms"], 1000.0)
        self.assertEqual(events[1]["result"]["result"]["assignments"], {"1000": 0})

    def test_execution_lifecycle_is_written_as_readable_log(self) -> None:
        async def scenario() -> None:
            queued_record = {
                "job_id": "job-readable",
                "status": "queued",
                "submitted_at": 123.0,
                "queue_position": 1,
                "request": {
                    "method": "POST",
                    "path": "/api/run",
                    "client_host": "127.0.0.1",
                    "user_agent": "test-agent",
                },
                "effective_config": {
                    "num_processes": 2,
                    "num_cores": 2,
                    "layers": 1,
                    "steps": 25,
                    "weights": [0.4, 0.6],
                },
            }
            completed_record = {
                **queued_record,
                "status": "done",
                "started_at": 124.0,
                "updated_at": 125.0,
                "result": {
                    "duration_ms": 1000.0,
                    "output_type": "SchedulingOutput",
                    "result": {
                        "load_imbalance": 0.2,
                        "validation": {"valid": True, "is_optimal": False},
                        "final_assignments": {"1000": 0, "1001": 1},
                    },
                },
            }

            await log_job_queued(queued_record)
            await log_job_completed(completed_record)

        asyncio.run(scenario())
        readable = settings.execution_log_path.read_text(encoding="utf-8")

        self.assertIn("execution.queued", readable)
        self.assertIn("job-readable", readable)
        self.assertIn("request: POST /api/run from 127.0.0.1", readable)
        self.assertIn("config: num_processes=2, num_cores=2, layers=1, steps=25", readable)
        self.assertIn("duration: 1.00s", readable)
        self.assertIn("output: SchedulingOutput", readable)
        self.assertIn("result: valid=True, optimal=False, load_imbalance=0.2", readable)
        self.assertIn("assignments: 1000->0, 1001->1", readable)

    def test_failed_execution_logs_exception_details(self) -> None:
        async def scenario() -> dict:
            try:
                raise RuntimeError("solver failed")
            except RuntimeError as exc:
                await log_job_failed(
                    {
                        "job_id": "job-2",
                        "status": "failed",
                        "submitted_at": 123.0,
                        "started_at": 124.0,
                        "updated_at": 125.0,
                        "effective_config": {"num_processes": 1},
                    },
                    exc,
                )
            return (await read_recent_events())[0]

        event = asyncio.run(scenario())

        self.assertEqual(event["event"], "execution.failed")
        self.assertEqual(event["error"]["type"], "RuntimeError")
        self.assertEqual(event["error"]["message"], "solver failed")
        self.assertIn("RuntimeError: solver failed", event["error"]["traceback"])

    def test_execution_log_rotates_by_age(self) -> None:
        old_timestamp = time.time() - (15 * 86_400)
        settings.execution_json_log_path.write_text(
            json.dumps(
                {
                    "event": "old",
                    "timestamp_unix": old_timestamp,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        asyncio.run(record_event("new"))

        archives = sorted(settings.execution_json_log_path.parent.glob("executions-*.jsonl"))
        self.assertEqual(len(archives), 1)
        self.assertIn('"event": "old"', archives[0].read_text(encoding="utf-8"))

        events = asyncio.run(read_recent_events())
        self.assertEqual([event["event"] for event in events], ["old", "new"])

    def test_execution_log_rotates_by_size_and_prunes_old_archives(self) -> None:
        object.__setattr__(settings, "execution_log_max_bytes", 1)
        object.__setattr__(settings, "execution_log_retention_files", 2)

        async def scenario() -> list[dict]:
            for index in range(5):
                await record_event("rotation.test", sequence=index)
            return await read_recent_events(10)

        events = asyncio.run(scenario())
        archives = sorted(settings.execution_json_log_path.parent.glob("executions-*.jsonl"))

        self.assertEqual(len(archives), 2)
        self.assertEqual(
            [event["sequence"] for event in events],
            [2, 3, 4],
        )
