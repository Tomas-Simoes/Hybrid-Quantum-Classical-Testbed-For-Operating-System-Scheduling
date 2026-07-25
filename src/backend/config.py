from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABSOLUTE_PUBLIC_MAX_N = 50
ABSOLUTE_PUBLIC_MAX_BUNDLES = 50
ABSOLUTE_PUBLIC_MAX_CORES = 4
ABSOLUTE_PUBLIC_MAX_QAOA_LAYERS = 3
ABSOLUTE_PUBLIC_MAX_QAOA_STEPS = 50
ABSOLUTE_PUBLIC_MAX_TOP_K = 32
ABSOLUTE_PUBLIC_MAX_QUBITS = 16
ABSOLUTE_PUBLIC_MAX_QUEUE_SIZE = 25
ABSOLUTE_PUBLIC_MAX_ACTIVE_JOBS = ABSOLUTE_PUBLIC_MAX_QUEUE_SIZE + 1
DEFAULT_PUBLIC_MAX_JOBS_PER_IP = 5
ABSOLUTE_PUBLIC_MAX_JOBS_PER_IP = ABSOLUTE_PUBLIC_MAX_ACTIVE_JOBS
ABSOLUTE_PUBLIC_MAX_JOB_TIMEOUT_SECONDS = 300


def _int_env(
    name: str,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    value = os.getenv(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError:
            parsed = default
    bounded = max(minimum, parsed)
    if maximum is not None:
        return min(maximum, bounded)
    return bounded


def _float_env(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    value = os.getenv(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value)
        except ValueError:
            parsed = default
    bounded = max(minimum, parsed)
    if maximum is not None:
        return min(maximum, bounded)
    return bounded


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


def _rate_limit_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _str_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class AdapterSettings:
    public_max_n: int = _int_env(
        "PUBLIC_MAX_N",
        ABSOLUTE_PUBLIC_MAX_N,
        maximum=ABSOLUTE_PUBLIC_MAX_N,
    )
    public_max_bundles: int = _int_env(
        "PUBLIC_MAX_BUNDLES",
        ABSOLUTE_PUBLIC_MAX_BUNDLES,
        maximum=ABSOLUTE_PUBLIC_MAX_BUNDLES,
    )
    public_max_cores: int = _int_env(
        "PUBLIC_MAX_CORES",
        4,
        maximum=ABSOLUTE_PUBLIC_MAX_CORES,
    )
    public_max_qaoa_layers: int = _int_env(
        "PUBLIC_MAX_QAOA_LAYERS",
        3,
        maximum=ABSOLUTE_PUBLIC_MAX_QAOA_LAYERS,
    )
    public_max_qaoa_steps: int = _int_env(
        "PUBLIC_MAX_QAOA_STEPS",
        50,
        maximum=ABSOLUTE_PUBLIC_MAX_QAOA_STEPS,
    )
    public_max_top_k: int = _int_env(
        "PUBLIC_MAX_TOP_K",
        32,
        maximum=ABSOLUTE_PUBLIC_MAX_TOP_K,
    )
    public_max_qubits: int = _int_env(
        "PUBLIC_MAX_QUBITS",
        16,
        maximum=ABSOLUTE_PUBLIC_MAX_QUBITS,
    )
    public_max_queue_size: int = _int_env(
        "PUBLIC_MAX_QUEUE_SIZE",
        25,
        maximum=ABSOLUTE_PUBLIC_MAX_QUEUE_SIZE,
    )
    public_max_jobs_per_ip: int = field(
        default_factory=lambda: _int_env(
            "PUBLIC_MAX_JOBS_PER_IP",
            DEFAULT_PUBLIC_MAX_JOBS_PER_IP,
            maximum=ABSOLUTE_PUBLIC_MAX_JOBS_PER_IP,
        )
    )
    public_max_active_jobs: int = _int_env(
        "PUBLIC_MAX_ACTIVE_JOBS",
        26,
        maximum=ABSOLUTE_PUBLIC_MAX_ACTIVE_JOBS,
    )
    public_job_timeout_seconds: int = _int_env(
        "PUBLIC_JOB_TIMEOUT_SECONDS",
        300,
        maximum=ABSOLUTE_PUBLIC_MAX_JOB_TIMEOUT_SECONDS,
    )
    public_max_request_bytes: int = _int_env("PUBLIC_MAX_REQUEST_BYTES", 20_000)
    public_job_ttl_seconds: int = _int_env("PUBLIC_JOB_TTL_SECONDS", 3_600)
    public_max_completed_jobs: int = _int_env("PUBLIC_MAX_COMPLETED_JOBS", 100)
    public_default_total_weight: float = _float_env("PUBLIC_DEFAULT_TOTAL_WEIGHT", 1.0)
    run_rate_limit: str = _rate_limit_env("RUN_RATE_LIMIT", "1/second")
    poll_rate_limit: str = _rate_limit_env("POLL_RATE_LIMIT", "120/minute")
    info_rate_limit: str = _rate_limit_env("INFO_RATE_LIMIT", "30/minute")
    bug_report_rate_limit: str = _rate_limit_env("BUG_REPORT_RATE_LIMIT", "3/hour")
    bug_report_min_seconds: int = _int_env("BUG_REPORT_MIN_SECONDS", 3, minimum=0, maximum=60)
    bug_report_max_seconds: int = _int_env(
        "BUG_REPORT_MAX_SECONDS",
        7_200,
        minimum=60,
        maximum=86_400,
    )
    bug_report_max_links: int = _int_env("BUG_REPORT_MAX_LINKS", 3, minimum=0, maximum=10)
    bug_report_duplicate_ttl_seconds: int = _int_env(
        "BUG_REPORT_DUPLICATE_TTL_SECONDS",
        3_600,
        minimum=60,
        maximum=86_400,
    )
    bug_report_log_path: Path = Path(
        os.getenv(
            "BUG_REPORT_LOG_PATH",
            str(PROJECT_ROOT / "logs" / "bug_reports.jsonl"),
        )
    )
    bug_report_recipient: str = _str_env("BUG_REPORT_TO", "tomas20simoes@gmail.com")
    bug_report_sender: str = _str_env("BUG_REPORT_FROM", "bug-reports@localhost")
    smtp_host: str = _str_env("SMTP_HOST")
    smtp_port: int = _int_env("SMTP_PORT", 587, minimum=1, maximum=65_535)
    smtp_username: str = _str_env("SMTP_USERNAME")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_starttls: bool = _bool_env("SMTP_STARTTLS", True)
    smtp_ssl: bool = _bool_env("SMTP_SSL", False)
    cors_allow_credentials: bool = _bool_env("CORS_ALLOW_CREDENTIALS", False)
    enable_api_docs: bool = _bool_env("ENABLE_API_DOCS", False)
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
    allowed_origins: list[str] = None
    allowed_hosts: list[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_origins", _origins_env())
        object.__setattr__(self, "allowed_hosts", _hosts_env())


settings = AdapterSettings()

PUBLIC_MAX_N = settings.public_max_n
PUBLIC_MAX_BUNDLES = settings.public_max_bundles
