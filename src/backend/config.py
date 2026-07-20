from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except ValueError:
        return default


def _origins_env() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class AdapterSettings:
    public_max_n: int = _int_env("PUBLIC_MAX_N", 10)
    public_max_bundles: int = _int_env("PUBLIC_MAX_BUNDLES", 10)
    public_max_cores: int = _int_env("PUBLIC_MAX_CORES", 4)
    public_max_qaoa_layers: int = _int_env("PUBLIC_MAX_QAOA_LAYERS", 3)
    public_max_qaoa_steps: int = _int_env("PUBLIC_MAX_QAOA_STEPS", 50)
    public_max_top_k: int = _int_env("PUBLIC_MAX_TOP_K", 32)
    public_max_queue_size: int = _int_env("PUBLIC_MAX_QUEUE_SIZE", 25)
    public_default_total_weight: float = _float_env("PUBLIC_DEFAULT_TOTAL_WEIGHT", 1.0)
    run_rate_limit: str = os.getenv("RUN_RATE_LIMIT", "1/second")
    allowed_origins: list[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_origins", _origins_env())


settings = AdapterSettings()

PUBLIC_MAX_N = settings.public_max_n
PUBLIC_MAX_BUNDLES = settings.public_max_bundles
