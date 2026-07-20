from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = SRC_ROOT / "core"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

import numpy as np

from data_contracts import (
    DecompositorConfig,
    ProcessInfo,
    QAOAConfig,
    QUBOConfig,
    SystemSnapshot,
    TracerConfig,
)
from decomposition.subqubo_heuristics import Heuristic
from main import SchedulingEngine

from .validation import RunConfig


def build_preset_snapshot(config: RunConfig) -> SystemSnapshot:
    return SystemSnapshot(
        timestamp=time.time(),
        num_cores=config.num_cores,
        processes=[
            ProcessInfo(
                pid=1000 + index,
                command=f"proc_{index}",
                cpu_weight=weight,
                current_core=0,
                rss_mb=weight * 1024,
                priority=20,
                io_wait_ratio=None,
                priority_class=None,
            )
            for index, weight in enumerate(config.weights or [])
        ],
        total_ram_mb=None,
        snapshot_id=None,
    )


def build_pipeline_configs(
    config: RunConfig,
) -> tuple[QAOAConfig, QUBOConfig, TracerConfig, DecompositorConfig, SystemSnapshot]:
    qaoa_cfg = QAOAConfig(
        layers=config.layers,
        steps=config.steps,
        learning_rate=config.learning_rate,
        top_k=config.top_k,
        mixer_type=config.mixer_type,
        init_gamma=config.init_gamma,
        init_beta=config.init_beta,
    )
    qubo_cfg = QUBOConfig(
        penalty=config.penalty,
        num_cores=config.num_cores,
        snapshot=None,
        target_load=config.target_load,
    )
    tracer_cfg = TracerConfig(
        min_rss=config.min_rss,
        min_cpu=config.min_cpu,
        cpu_interval=config.cpu_interval,
        num_samples=config.num_samples,
        live_mode=False,
    )
    decompositor_cfg = DecompositorConfig(
        qubit_max=int(config.qubit_max or config.num_cores),
        num_cores=config.num_cores,
        io_alpha=config.io_alpha,
        affinity_alpha=config.affinity_alpha,
        homogeneity_threshold=config.homogeneity_threshold,
        zscore_threshold=config.zscore_threshold,
        sorting_strategy=Heuristic[config.sorting_strategy],
    )
    return qaoa_cfg, qubo_cfg, tracer_cfg, decompositor_cfg, build_preset_snapshot(config)


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    qaoa_cfg, qubo_cfg, tracer_cfg, decompositor_cfg, preset_snapshot = build_pipeline_configs(config)
    started = time.perf_counter()
    output = SchedulingEngine.run_job(
        qaoa_cfg=qaoa_cfg,
        qubo_cfg=qubo_cfg,
        tracer_cfg=tracer_cfg,
        decompositor_cfg=decompositor_cfg,
        preset_snapshot=preset_snapshot,
    )
    return {
        "output_type": type(output).__name__,
        "duration_ms": (time.perf_counter() - started) * 1000,
        "effective_config": config.model_dump(),
        "result": jsonable(output),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
