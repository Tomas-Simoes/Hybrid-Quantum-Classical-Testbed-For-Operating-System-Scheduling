from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..config import settings
from ..ratelimit import limiter

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCALABILITY_RESULTS = PROJECT_ROOT / "precomputed" / "scalability_results.json"


@router.get("/scalability")
@limiter.limit(settings.info_rate_limit)
async def scalability(request: Request) -> dict:
    if not SCALABILITY_RESULTS.exists():
        raise HTTPException(status_code=404, detail="Precomputed scalability results are not available.")
    return json.loads(SCALABILITY_RESULTS.read_text(encoding="utf-8"))
