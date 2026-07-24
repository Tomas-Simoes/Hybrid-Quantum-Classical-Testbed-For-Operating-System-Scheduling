from __future__ import annotations

import asyncio
import unittest

from src.backend import job_store
from src.backend.config import settings
from src.backend.validation import RunConfig


class BackendJobStoreTests(unittest.TestCase):
    def tearDown(self) -> None:
        async def clear_jobs() -> None:
            async with job_store._lock:
                job_store._jobs.clear()

        asyncio.run(clear_jobs())

    def test_public_queue_metadata_and_positions_are_refreshed(self) -> None:
        async def scenario() -> tuple[dict, dict, dict]:
            config = RunConfig(num_processes=2, weights=[0.4, 0.6])
            await job_store.create_job("job-1", config, 1)
            await job_store.create_job("job-2", config, 2)
            await job_store.create_job("job-3", config, 3)

            running = await job_store.mark_running("job-1")
            second = await job_store.get_job("job-2")
            third = await job_store.get_job("job-3")
            return running, second, third

        running, second, third = asyncio.run(scenario())

        self.assertEqual(running["queue_position"], 0)
        self.assertEqual(second["queue_position"], 1)
        self.assertEqual(third["queue_position"], 2)
        self.assertEqual(second["queue_capacity"], settings.public_max_queue_size)
        self.assertEqual(second["queue_running_count"], 1)
