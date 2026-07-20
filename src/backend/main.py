from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from .config import settings
from .queue import start_worker, stop_worker
from .ratelimit import limiter
from .routes import health, run, scalability


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_worker()
    yield
    await stop_worker()


app = FastAPI(title="Hybrid Scheduler Adapter", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(run.router, prefix="/api")
app.include_router(scalability.router, prefix="/api")
