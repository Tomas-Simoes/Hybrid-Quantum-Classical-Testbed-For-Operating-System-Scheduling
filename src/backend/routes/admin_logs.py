from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse

from ..config import settings
from ..execution_log import read_recent_events, read_recent_text
from ..feedback import read_recent_bug_report_records, read_recent_bug_report_text
from ..ratelimit import limiter

router = APIRouter(include_in_schema=False)

_MIN_ADMIN_LOG_TOKEN_LENGTH = 32


def _raise_invalid_credentials() -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def authorize_admin_log_access(authorization: str | None) -> None:
    expected = settings.admin_log_token.strip()
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    if len(expected) < _MIN_ADMIN_LOG_TOKEN_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin log access is not safely configured.",
        )

    scheme, _, provided = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not provided.strip():
        _raise_invalid_credentials()
    if not secrets.compare_digest(provided.strip(), expected):
        _raise_invalid_credentials()


def require_admin_log_token(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    authorize_admin_log_access(authorization)


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@router.get("/admin/execution-logs")
@limiter.limit(settings.info_rate_limit)
async def get_execution_logs(
    request: Request,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    _: None = Depends(require_admin_log_token),
) -> dict:
    _no_store(response)
    events = await read_recent_events(limit)
    return {"count": len(events), "events": events}


@router.get("/admin/execution-logs.txt", response_class=PlainTextResponse)
@limiter.limit(settings.info_rate_limit)
async def get_execution_logs_text(
    request: Request,
    max_chars: Annotated[int, Query(ge=1_000, le=200_000)] = 50_000,
    _: None = Depends(require_admin_log_token),
) -> PlainTextResponse:
    text = await read_recent_text(max_chars)
    return PlainTextResponse(
        text,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.get("/admin/bug-logs")
@limiter.limit(settings.info_rate_limit)
async def get_bug_logs(
    request: Request,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    _: None = Depends(require_admin_log_token),
) -> dict:
    _no_store(response)
    records = await read_recent_bug_report_records(limit)
    return {"count": len(records), "reports": records}


@router.get("/admin/bug-logs.txt", response_class=PlainTextResponse)
@limiter.limit(settings.info_rate_limit)
async def get_bug_logs_text(
    request: Request,
    max_chars: Annotated[int, Query(ge=1_000, le=200_000)] = 50_000,
    _: None = Depends(require_admin_log_token),
) -> PlainTextResponse:
    text = await read_recent_bug_report_text(max_chars)
    return PlainTextResponse(
        text,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )
