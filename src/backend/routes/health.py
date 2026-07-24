from __future__ import annotations

from fastapi import APIRouter, Request

from ..config import settings
from ..ratelimit import limiter

router = APIRouter()


@router.get("/health")
@limiter.limit(settings.info_rate_limit)
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}
