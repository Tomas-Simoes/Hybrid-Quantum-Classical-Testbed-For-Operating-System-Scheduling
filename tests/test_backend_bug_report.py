from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.backend.config import settings
from src.backend.feedback import (
    BugReport,
    bug_report_rejection_reason,
    clear_recent_report_fingerprints,
    mark_duplicate_bug_report,
    read_recent_bug_report_records,
    read_recent_bug_report_text,
    smtp_is_configured,
    write_bug_report_log,
)


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


def make_report(now: float = 1_700_000_000.0, **overrides: object) -> BugReport:
    payload = {
        "subject": "Broken result chart",
        "message": "The result chart does not render after a successful run.",
        "severity": "medium",
        "page_url": "https://scheduler.example/chamber",
        "form_started_at": int((now - 5) * 1_000),
    }
    payload.update(overrides)
    return BugReport(**payload)


class BackendBugReportTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_recent_report_fingerprints()

    def test_requires_valid_email_when_reply_is_requested(self) -> None:
        with self.assertRaises(ValidationError):
            make_report(email="not-an-email", contact_consent=True)

        with self.assertRaises(ValidationError):
            make_report(email="", contact_consent=True)

    def test_spam_guards_cover_honeypot_timing_and_links(self) -> None:
        now = 1_700_000_000.0
        with SettingsPatch(
            bug_report_min_seconds=3,
            bug_report_max_seconds=7_200,
            bug_report_max_links=1,
        ):
            self.assertEqual(
                bug_report_rejection_reason(make_report(now=now, website="filled"), now=now),
                "honeypot",
            )
            self.assertEqual(
                bug_report_rejection_reason(
                    make_report(now=now, form_started_at=int((now - 1) * 1_000)),
                    now=now,
                ),
                "too_fast",
            )
            self.assertEqual(
                bug_report_rejection_reason(
                    make_report(now=now, form_started_at=int((now - 8_000) * 1_000)),
                    now=now,
                ),
                "stale",
            )
            self.assertEqual(
                bug_report_rejection_reason(make_report(now=now, form_started_at=None), now=now),
                "missing_timer",
            )
            self.assertEqual(
                bug_report_rejection_reason(
                    make_report(
                        now=now,
                        message=(
                            "The page links to http://one.example and http://two.example "
                            "while failing to render the console."
                        ),
                    ),
                    now=now,
                ),
                "too_many_links",
            )

    def test_duplicate_reports_are_suppressed_temporarily(self) -> None:
        first = make_report()
        second = make_report()

        with SettingsPatch(bug_report_duplicate_ttl_seconds=60):
            self.assertFalse(mark_duplicate_bug_report(first, now=100.0))
            self.assertTrue(mark_duplicate_bug_report(second, now=101.0))
            self.assertFalse(mark_duplicate_bug_report(second, now=161.0))

    def test_smtp_delivery_is_opt_in(self) -> None:
        with SettingsPatch(
            smtp_host="",
            bug_report_recipient="owner@example.com",
            bug_report_sender="reports@example.com",
        ):
            self.assertFalse(smtp_is_configured())

        with SettingsPatch(
            smtp_host="smtp.example.com",
            bug_report_recipient="owner@example.com",
            bug_report_sender="reports@example.com",
        ):
            self.assertTrue(smtp_is_configured())

    def test_report_log_excludes_honeypot_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "bug_reports.jsonl"
            with SettingsPatch(bug_report_log_path=log_path):
                write_bug_report_log(
                    make_report(website="should-not-be-written"),
                    {
                        "client_host": "203.0.113.10",
                        "user_agent": "unit-test",
                        "origin": "https://scheduler.example",
                        "referer": "https://scheduler.example/#contacts",
                    },
                    "logged",
                )

            record = json.loads(log_path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["delivery"], "logged")
        self.assertEqual(record["metadata"]["client_host"], "203.0.113.10")
        self.assertEqual(record["report"]["subject"], "Broken result chart")
        self.assertNotIn("website", record["report"])

    def test_report_log_can_be_read_as_recent_records_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "bug_reports.jsonl"
            with SettingsPatch(bug_report_log_path=log_path, execution_log_max_read=50):
                write_bug_report_log(make_report(subject="First broken chart"), {}, "logged")
                write_bug_report_log(make_report(subject="Second broken chart"), {}, "logged")

                records = asyncio.run(read_recent_bug_report_records(limit=1))
                text = asyncio.run(read_recent_bug_report_text(max_chars=1_000))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["report"]["subject"], "Second broken chart")
        self.assertIn("First broken chart", text)
        self.assertIn("Second broken chart", text)
