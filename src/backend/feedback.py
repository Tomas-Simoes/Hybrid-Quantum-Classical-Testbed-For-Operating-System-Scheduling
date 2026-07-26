from __future__ import annotations

import asyncio
import hashlib
import json
import re
import smtplib
import ssl
import threading
import time
from collections import deque
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import settings


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMAIL_RE = re.compile(r"^[^\s@<>]{1,64}@[^\s@<>]{1,253}\.[^\s@<>]{2,}$")
_LINK_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_recent_report_fingerprints: dict[str, float] = {}
_fingerprint_lock = threading.Lock()
_bug_log_read_lock = asyncio.Lock()


class BugReport(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    subject: str = Field(min_length=4, max_length=120)
    message: str = Field(min_length=20, max_length=4_000)
    severity: Literal["low", "medium", "high"] = "medium"
    name: str = Field(default="", max_length=80)
    email: str = Field(default="", max_length=254)
    page_url: str = Field(default="", max_length=2_048)
    steps: str = Field(default="", max_length=1_600)
    expected: str = Field(default="", max_length=1_200)
    actual: str = Field(default="", max_length=1_200)
    contact_consent: bool = False
    website: str = Field(default="", max_length=120)
    form_started_at: int | None = Field(default=None, gt=0)

    @field_validator(
        "subject",
        "message",
        "name",
        "email",
        "page_url",
        "steps",
        "expected",
        "actual",
        "website",
        mode="before",
    )
    @classmethod
    def clean_text(cls, value: object) -> str:
        if value is None:
            return ""
        return _CONTROL_RE.sub("", str(value)).strip()

    @field_validator("email")
    @classmethod
    def validate_optional_email(cls, value: str) -> str:
        if value and not _EMAIL_RE.match(value):
            raise ValueError("Enter a valid email address.")
        return value

    @field_validator("page_url")
    @classmethod
    def validate_optional_page_url(cls, value: str) -> str:
        if not value:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Use a valid http or https page URL.")
        if parsed.username or parsed.password:
            raise ValueError("Page URL must not include credentials.")
        return value

    @model_validator(mode="after")
    def validate_contact_preferences(self) -> BugReport:
        if self.contact_consent and not self.email:
            raise ValueError("Email is required when requesting a reply.")
        return self


class BugReportDeliveryError(RuntimeError):
    """Raised when a configured email delivery backend fails."""


def clear_recent_report_fingerprints() -> None:
    with _fingerprint_lock:
        _recent_report_fingerprints.clear()


def _submitted_age_seconds(report: BugReport, now: float) -> float | None:
    if report.form_started_at is None:
        return None
    started = float(report.form_started_at)
    if started > 10_000_000_000:
        started = started / 1_000
    return now - started


def bug_report_rejection_reason(report: BugReport, now: float | None = None) -> str | None:
    now = time.time() if now is None else now
    if report.website:
        return "honeypot"

    age = _submitted_age_seconds(report, now)
    if age is None:
        return "missing_timer"
    if age < settings.bug_report_min_seconds:
        return "too_fast"
    if age > settings.bug_report_max_seconds:
        return "stale"

    report_text = "\n".join(
        [
            report.subject,
            report.message,
            report.page_url,
            report.steps,
            report.expected,
            report.actual,
        ]
    )
    if len(_LINK_RE.findall(report_text)) > settings.bug_report_max_links:
        return "too_many_links"

    return None


def mark_duplicate_bug_report(report: BugReport, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    fingerprint = _report_fingerprint(report)
    expires_at = now + settings.bug_report_duplicate_ttl_seconds

    with _fingerprint_lock:
        expired = [
            key
            for key, expiry in _recent_report_fingerprints.items()
            if expiry <= now
        ]
        for key in expired:
            _recent_report_fingerprints.pop(key, None)

        if fingerprint in _recent_report_fingerprints:
            return True
        _recent_report_fingerprints[fingerprint] = expires_at
        return False


def smtp_is_configured() -> bool:
    return bool(settings.smtp_host and settings.bug_report_recipient and settings.bug_report_sender)


def send_bug_report_email(report: BugReport, metadata: dict[str, str | None]) -> None:
    if not smtp_is_configured():
        return

    message = EmailMessage()
    message["Subject"] = _safe_header(f"[Hybrid Scheduler] Bug report: {report.subject}")
    message["From"] = _safe_header(settings.bug_report_sender)
    message["To"] = _safe_header(settings.bug_report_recipient)
    if report.contact_consent and report.email:
        message["Reply-To"] = _safe_header(report.email)
    message.set_content(_format_email_body(report, metadata))

    context = ssl.create_default_context()
    try:
        if settings.smtp_ssl:
            with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=10, context=context) as smtp:
                _smtp_login_if_needed(smtp)
                smtp.send_message(message)
            return

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            if settings.smtp_starttls:
                smtp.starttls(context=context)
            _smtp_login_if_needed(smtp)
            smtp.send_message(message)
    except Exception as exc:  # pragma: no cover - exact SMTP failures are provider-specific.
        raise BugReportDeliveryError("Bug report email delivery failed.") from exc


def write_bug_report_log(
    report: BugReport,
    metadata: dict[str, str | None],
    delivery: Literal["logged", "emailed", "email_failed"],
) -> None:
    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "delivery": delivery,
        "metadata": {
            "client_host": _truncate(metadata.get("client_host"), 128),
            "user_agent": _truncate(metadata.get("user_agent"), 512),
            "origin": _truncate(metadata.get("origin"), 512),
            "referer": _truncate(metadata.get("referer"), 512),
        },
        "report": report.model_dump(exclude={"website"}),
    }
    settings.bug_report_log_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.bug_report_log_path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True)
        handle.write("\n")


def _read_recent_bug_report_records(limit: int) -> list[dict]:
    records: deque[dict] = deque(maxlen=limit)
    path = settings.bug_report_log_path
    if not path.exists() or not path.is_file():
        return []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                records.append(
                    {
                        "event": "bug_report_log.invalid_line",
                        "line_number": line_number,
                        "raw": stripped,
                    }
                )
    return list(records)


async def read_recent_bug_report_records(limit: int | None = None) -> list[dict]:
    requested_limit = limit or settings.execution_log_max_read
    bounded_limit = max(1, min(int(requested_limit), settings.execution_log_max_read))
    async with _bug_log_read_lock:
        return await asyncio.to_thread(_read_recent_bug_report_records, bounded_limit)


def _read_bug_report_text_tail(max_chars: int) -> str:
    path = settings.bug_report_log_path
    if not path.exists() or not path.is_file():
        return ""

    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


async def read_recent_bug_report_text(max_chars: int = 50_000) -> str:
    bounded_max_chars = max(1_000, min(int(max_chars), 200_000))
    async with _bug_log_read_lock:
        return await asyncio.to_thread(_read_bug_report_text_tail, bounded_max_chars)


def _report_fingerprint(report: BugReport) -> str:
    normalized = "\n".join(
        [
            report.email.lower(),
            report.subject.lower(),
            report.message.lower(),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_header(value: str) -> str:
    return re.sub(r"[\r\n]+", " ", value).strip()[:240]


def _truncate(value: str | None, maximum: int) -> str | None:
    if value is None:
        return None
    return value[:maximum]


def _smtp_login_if_needed(smtp: smtplib.SMTP) -> None:
    if settings.smtp_username and settings.smtp_password:
        smtp.login(settings.smtp_username, settings.smtp_password)


def _format_email_body(report: BugReport, metadata: dict[str, str | None]) -> str:
    sections = [
        ("Subject", report.subject),
        ("Severity", report.severity),
        ("Name", report.name or "Not provided"),
        ("Email", report.email or "Not provided"),
        ("May reply", "yes" if report.contact_consent else "no"),
        ("Page", report.page_url or "Not provided"),
        ("Client", metadata.get("client_host") or "unknown"),
        ("User agent", metadata.get("user_agent") or "unknown"),
        ("Message", report.message),
        ("Steps", report.steps or "Not provided"),
        ("Expected", report.expected or "Not provided"),
        ("Actual", report.actual or "Not provided"),
    ]
    return "\n\n".join(f"{label}:\n{value}" for label, value in sections)
