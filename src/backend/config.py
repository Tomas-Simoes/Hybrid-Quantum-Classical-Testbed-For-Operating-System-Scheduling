from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _origins_env() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _hosts_env() -> list[str]:
    raw = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")
    return [host.strip() for host in raw.split(",") if host.strip()]


@dataclass(frozen=True)
class AdapterSettings:
    public_max_n: int = _int_env("PUBLIC_MAX_N", 10)
    public_max_bundles: int = _int_env("PUBLIC_MAX_BUNDLES", 10)
    public_max_cores: int = _int_env("PUBLIC_MAX_CORES", 4)
    public_max_qaoa_layers: int = _int_env("PUBLIC_MAX_QAOA_LAYERS", 3)
    public_max_qaoa_steps: int = _int_env("PUBLIC_MAX_QAOA_STEPS", 50)
    public_max_top_k: int = _int_env("PUBLIC_MAX_TOP_K", 32)
    public_max_qubits: int = _int_env("PUBLIC_MAX_QUBITS", 16)
    public_max_queue_size: int = _int_env("PUBLIC_MAX_QUEUE_SIZE", 25)
    public_max_request_bytes: int = _int_env("PUBLIC_MAX_REQUEST_BYTES", 20_000)
    public_job_ttl_seconds: int = _int_env("PUBLIC_JOB_TTL_SECONDS", 3_600)
    public_max_completed_jobs: int = _int_env("PUBLIC_MAX_COMPLETED_JOBS", 100)
    public_default_total_weight: float = _float_env("PUBLIC_DEFAULT_TOTAL_WEIGHT", 1.0)
    run_rate_limit: str = os.getenv("RUN_RATE_LIMIT", "1/second")
    cors_allow_credentials: bool = _bool_env("CORS_ALLOW_CREDENTIALS", False)
    execution_log_path: Path = Path(
        os.getenv(
            "EXECUTION_LOG_PATH",
            str(PROJECT_ROOT / "logs" / "backend_executions.log"),
        )
    )
    execution_json_log_path: Path = Path(
        os.getenv(
            "EXECUTION_JSON_LOG_PATH",
            str(PROJECT_ROOT / "logs" / "backend_executions.jsonl"),
        )
    )
    execution_log_max_read: int = _int_env("EXECUTION_LOG_MAX_READ", 500, minimum=1)
    execution_log_rotation_days: int = _int_env(
        "EXECUTION_LOG_ROTATION_DAYS",
        14,
        minimum=1,
    )
    execution_log_max_bytes: int = _int_env(
        "EXECUTION_LOG_MAX_BYTES",
        10_000_000,
        minimum=1_024,
    )
    execution_log_retention_files: int = _int_env(
        "EXECUTION_LOG_RETENTION_FILES",
        6,
        minimum=1,
    )
    expose_execution_logs: bool = _bool_env("EXPOSE_EXECUTION_LOGS", False)
    allowed_origins: list[str] = None
    allowed_hosts: list[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_origins", _origins_env())
        object.__setattr__(self, "allowed_hosts", _hosts_env())


settings = AdapterSettings()

PUBLIC_MAX_N = settings.public_max_n
PUBLIC_MAX_BUNDLES = settings.public_max_bundles
