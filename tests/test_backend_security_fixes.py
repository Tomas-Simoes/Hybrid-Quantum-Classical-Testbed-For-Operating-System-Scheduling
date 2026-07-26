from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from src.backend import job_store
from src.backend import queue as job_queue
from src.backend.config import (
    ABSOLUTE_PUBLIC_MAX_JOB_TIMEOUT_SECONDS,
    ABSOLUTE_PUBLIC_MAX_JOBS_PER_IP,
    ABSOLUTE_PUBLIC_MAX_N,
    AdapterSettings,
    settings,
)
from src.backend.main import app
from src.backend.routes import admin_logs
from src.backend.validation import RunConfig


def _sleeping_pipeline_child(config_data: dict, result_path: str) -> None:
    time.sleep(2)


class SettingsPatch:
    def __init__(self, **values: object) -> None:
        self.values = values
        self.originals: dict[str, object] = {}

    def __enter__(self) -> SettingsPatch:
        for name, value in self.values.items():
            self.originals[name] = getattr(settings, name)
            object.__setattr__(settings, name, value)
        return self

    def __exit__(self, *args: object) -> None:
        for name, value in self.originals.items():
            object.__setattr__(settings, name, value)


class BackendSecurityFixTests(unittest.TestCase):
    def tearDown(self) -> None:
        async def clear_state() -> None:
            await job_queue.stop_worker()
            async with job_store._lock:
                job_store._jobs.clear()

        asyncio.run(clear_state())
        job_queue._queue = None
        job_queue._admission_lock = asyncio.Lock()

    def test_public_runtime_caps_are_hard_limited(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "PUBLIC_MAX_N": "999",
                "PUBLIC_MAX_JOBS_PER_IP": "99",
                "PUBLIC_JOB_TIMEOUT_SECONDS": "999",
            },
        ):
            env_settings = AdapterSettings()

        self.assertEqual(env_settings.public_max_n, ABSOLUTE_PUBLIC_MAX_N)
        self.assertEqual(
            env_settings.public_max_jobs_per_ip,
            ABSOLUTE_PUBLIC_MAX_JOBS_PER_IP,
        )
        self.assertEqual(
            env_settings.public_job_timeout_seconds,
            ABSOLUTE_PUBLIC_MAX_JOB_TIMEOUT_SECONDS,
        )

        config = RunConfig(num_processes=999)
        self.assertEqual(config.num_processes, ABSOLUTE_PUBLIC_MAX_N)
        self.assertEqual(len(config.weights), ABSOLUTE_PUBLIC_MAX_N)

    def test_core_balance_is_not_publicly_accepted(self) -> None:
        with self.assertRaises(ValidationError):
            RunConfig(sorting_strategy="CORE_BALANCE")

    def test_public_log_and_auto_docs_routes_are_not_exposed(self) -> None:
        paths = {route.path for route in app.routes if hasattr(route, "path")}

        self.assertNotIn("/api/execution-logs", paths)
        self.assertNotIn("/api/execution-logs.txt", paths)
        self.assertIsNone(app.docs_url)
        self.assertIsNone(app.redoc_url)
        self.assertIsNone(app.openapi_url)

    def test_admin_log_routes_are_hidden_from_openapi(self) -> None:
        admin_paths = {
            "/admin/execution-logs",
            "/admin/execution-logs.txt",
            "/admin/bug-logs",
            "/admin/bug-logs.txt",
        }
        matching_routes = list(admin_logs.router.routes)

        self.assertEqual({route.path for route in matching_routes}, admin_paths)
        self.assertTrue(all(route.include_in_schema is False for route in matching_routes))

    def test_admin_log_auth_fails_closed_when_not_configured(self) -> None:
        with SettingsPatch(admin_log_token=""):
            with self.assertRaises(HTTPException) as context:
                admin_logs.authorize_admin_log_access(None)

        self.assertEqual(context.exception.status_code, 404)

    def test_admin_log_auth_rejects_unsafe_short_token(self) -> None:
        with SettingsPatch(admin_log_token="short"):
            with self.assertRaises(HTTPException) as context:
                admin_logs.authorize_admin_log_access("Bearer short")

        self.assertEqual(context.exception.status_code, 503)

    def test_admin_log_auth_requires_exact_bearer_token(self) -> None:
        token = "a" * 32
        with SettingsPatch(admin_log_token=token):
            with self.assertRaises(HTTPException) as context:
                admin_logs.authorize_admin_log_access(f"Bearer {token[:-1]}b")
            admin_logs.authorize_admin_log_access(f"Bearer {token}")

        self.assertEqual(context.exception.status_code, 401)

    def test_request_body_limit_rejects_stream_without_content_length(self) -> None:
        async def scenario() -> tuple[int, bytes, dict[bytes, bytes]]:
            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/bug-report",
                "raw_path": b"/api/bug-report",
                "query_string": b"",
                "root_path": "",
                "headers": [
                    (b"host", b"testserver"),
                    (b"content-type", b"application/json"),
                    (b"origin", b"http://localhost:5173"),
                ],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
            }
            request_messages = [
                {"type": "http.request", "body": b"12345", "more_body": True},
                {"type": "http.request", "body": b"678901", "more_body": False},
            ]
            sent_messages = []

            async def receive() -> dict:
                if request_messages:
                    return request_messages.pop(0)
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message: dict) -> None:
                sent_messages.append(message)

            await app(scope, receive, send)
            response_start = next(
                message for message in sent_messages if message["type"] == "http.response.start"
            )
            response_body = b"".join(
                message.get("body", b"")
                for message in sent_messages
                if message["type"] == "http.response.body"
            )
            headers = {name.lower(): value for name, value in response_start["headers"]}
            return response_start["status"], response_body, headers

        with SettingsPatch(public_max_request_bytes=10):
            status_code, body, headers = asyncio.run(scenario())

        self.assertEqual(status_code, 413)
        self.assertIn(b"Request body is too large", body)
        self.assertEqual(headers[b"access-control-allow-origin"], b"http://localhost:5173")
        self.assertEqual(headers[b"x-content-type-options"], b"nosniff")

    def test_enqueue_rejects_per_ip_active_job_limit(self) -> None:
        async def scenario() -> None:
            config = RunConfig(num_processes=1, num_cores=1, weights=[1.0])
            await job_queue.enqueue_run(config, {"client_host": "203.0.113.10"})

            with self.assertRaises(job_queue.ClientJobLimitError):
                await job_queue.enqueue_run(config, {"client_host": "203.0.113.10"})

            counts = await job_store.active_counts("203.0.113.10")
            self.assertEqual(counts["client_active"], 1)

        with SettingsPatch(public_max_jobs_per_ip=1):
            asyncio.run(scenario())

    def test_enqueue_allows_multiple_queued_jobs_from_same_client(self) -> None:
        async def scenario() -> None:
            config = RunConfig(num_processes=1, num_cores=1, weights=[1.0])
            first = await job_queue.enqueue_run(config, {"client_host": "203.0.113.10"})
            second = await job_queue.enqueue_run(config, {"client_host": "203.0.113.10"})
            third = await job_queue.enqueue_run(config, {"client_host": "203.0.113.10"})

            counts = await job_store.active_counts("203.0.113.10")
            self.assertEqual(counts["client_active"], 3)
            self.assertEqual(counts["client_queued"], 3)
            self.assertEqual(
                [
                    first["queue_position"],
                    second["queue_position"],
                    third["queue_position"],
                ],
                [1, 2, 3],
            )

        with SettingsPatch(
            public_max_queue_size=5,
            public_max_active_jobs=6,
            public_max_jobs_per_ip=5,
        ):
            job_queue._queue = None
            asyncio.run(scenario())

    def test_enqueue_rejects_full_queue_without_extra_job_record(self) -> None:
        async def scenario() -> None:
            config = RunConfig(num_processes=1, num_cores=1, weights=[1.0])
            await job_queue.enqueue_run(config, {"client_host": "203.0.113.10"})

            with self.assertRaises(job_queue.QueueFullError):
                await job_queue.enqueue_run(config, {"client_host": "203.0.113.11"})

            counts = await job_store.active_counts()
            self.assertEqual(counts["queued"], 1)
            self.assertEqual(counts["active"], 1)

        with SettingsPatch(
            public_max_queue_size=1,
            public_max_active_jobs=2,
            public_max_jobs_per_ip=2,
        ):
            job_queue._queue = None
            asyncio.run(scenario())

    def test_enqueue_preserves_fifo_queue_order(self) -> None:
        async def scenario() -> tuple[str, str, str, str]:
            config = RunConfig(num_processes=1, num_cores=1, weights=[1.0])
            first = await job_queue.enqueue_run(config, {"client_host": "203.0.113.10"})
            second = await job_queue.enqueue_run(config, {"client_host": "203.0.113.11"})

            queue = job_queue._job_queue()
            first_queued = queue.get_nowait()
            second_queued = queue.get_nowait()
            queue.task_done()
            queue.task_done()
            return (
                first["job_id"],
                second["job_id"],
                first_queued.job_id,
                second_queued.job_id,
            )

        first_id, second_id, first_queued_id, second_queued_id = asyncio.run(scenario())

        self.assertEqual(first_queued_id, first_id)
        self.assertEqual(second_queued_id, second_id)

    def test_pipeline_process_timeout_terminates_child(self) -> None:
        config = RunConfig(num_processes=1, num_cores=1, weights=[1.0])
        with patch.object(job_queue, "_pipeline_child", _sleeping_pipeline_child):
            with self.assertRaises(job_queue.JobTimeoutError):
                job_queue._run_pipeline_process(config, timeout_seconds=0.05)
