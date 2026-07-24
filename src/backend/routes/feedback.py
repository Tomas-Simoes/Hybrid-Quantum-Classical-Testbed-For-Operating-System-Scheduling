from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..feedback import (
    BugReport,
    BugReportDeliveryError,
    bug_report_rejection_reason,
    mark_duplicate_bug_report,
    send_bug_report_email,
    smtp_is_configured,
    write_bug_report_log,
)
from ..ratelimit import limiter

router = APIRouter()


_REJECTION_DETAILS = {
    "missing_timer": "Refresh the page and try again.",
    "too_fast": "Please wait a moment and try again.",
    "stale": "Refresh the page and submit the report again.",
    "too_many_links": "Bug reports can include only a few links.",
}


def _request_metadata(request: Request) -> dict[str, str | None]:
    return {
        "client_host": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "origin": request.headers.get("origin"),
        "referer": request.headers.get("referer"),
    }


def _enforce_allowed_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin not in settings.allowed_origins:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Origin is not allowed.")


@router.post("/bug-report", status_code=status.HTTP_201_CREATED)
@limiter.limit(settings.bug_report_rate_limit)
async def create_bug_report(request: Request, report: BugReport) -> dict[str, str]:
    _enforce_allowed_origin(request)

    metadata = _request_metadata(request)
    rejection_reason = bug_report_rejection_reason(report)
    if rejection_reason == "honeypot":
        return {"status": "received"}
    if rejection_reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_REJECTION_DETAILS[rejection_reason],
        )

    if mark_duplicate_bug_report(report):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Please wait before sending the same report again.",
        )

    delivery = "logged"
    if smtp_is_configured():
        try:
            await run_in_threadpool(send_bug_report_email, report, metadata)
            delivery = "emailed"
        except BugReportDeliveryError:
            delivery = "email_failed"

    await run_in_threadpool(write_bug_report_log, report, metadata, delivery)
    return {"status": "received"}
