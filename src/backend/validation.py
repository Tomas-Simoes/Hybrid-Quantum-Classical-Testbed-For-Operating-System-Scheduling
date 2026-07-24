from __future__ import annotations

import math
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import settings


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return min(max(value, minimum), maximum)


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _finite_float(value: float | int | None, name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, allow_inf_nan=False)

    num_processes: int = Field(
        default=2,
        validation_alias=AliasChoices("num_processes", "n_processes"),
    )
    num_cores: int = 2
    weights: list[float] | None = Field(default=None, max_length=settings.public_max_n)
    total_weight: float = 1.0

    penalty: float = 5.0
    target_load: float | None = None

    layers: int = 1
    steps: int = 25
    learning_rate: float = 0.05
    top_k: int = 10
    mixer_type: Literal["xy", "x"] = Field(
        default="xy",
        validation_alias=AliasChoices("mixer_type", "mixer"),
    )
    init_gamma: float = 0.5
    init_beta: float = 0.5

    qubit_max: int | None = None
    io_alpha: float = 0.5
    affinity_alpha: float = 0.8
    homogeneity_threshold: float = 0.3
    zscore_threshold: float = 1.5
    sorting_strategy: Literal[
        "WEIGHT_DESCENDING",
        "COUPLING_DESCENDING",
    ] = "WEIGHT_DESCENDING"

    min_rss: float = 20.0
    min_cpu: float = 0.005
    cpu_interval: int = 1
    num_samples: int = 3

    @field_validator("mixer_type", mode="before")
    @classmethod
    def normalize_mixer(cls, value: str) -> str:
        return str(value).strip().lower()

    @field_validator("sorting_strategy", mode="before")
    @classmethod
    def normalize_sorting_strategy(cls, value: str) -> str:
        return str(value).strip().upper()

    @model_validator(mode="after")
    def clamp_public_values(self) -> RunConfig:
        self.num_processes = _clamp_int(int(self.num_processes), 1, settings.public_max_n)
        self.num_cores = _clamp_int(int(self.num_cores), 1, settings.public_max_cores)
        self.total_weight = _clamp_float(_finite_float(self.total_weight, "total_weight"), 0.001, 100.0)

        if self.weights:
            weights = [
                _clamp_float(_finite_float(weight, "weights"), 0.0, 100.0)
                for weight in self.weights[: self.num_processes]
            ]
            if len(weights) < self.num_processes:
                remaining = self.num_processes - len(weights)
                filler = self.total_weight / self.num_processes
                weights.extend([filler] * remaining)
            self.weights = weights
        else:
            weight = settings.public_default_total_weight / self.num_processes
            self.weights = [weight] * self.num_processes

        self.penalty = _clamp_float(_finite_float(self.penalty, "penalty"), 0.001, 1_000.0)
        if self.target_load is not None:
            self.target_load = _clamp_float(_finite_float(self.target_load, "target_load"), 0.0, 100.0)

        self.layers = _clamp_int(int(self.layers), 1, settings.public_max_qaoa_layers)
        self.steps = _clamp_int(int(self.steps), 1, settings.public_max_qaoa_steps)
        self.learning_rate = _clamp_float(_finite_float(self.learning_rate, "learning_rate"), 0.0001, 1.0)
        self.top_k = _clamp_int(int(self.top_k), 1, settings.public_max_top_k)
        self.init_gamma = _clamp_float(_finite_float(self.init_gamma, "init_gamma"), 0.0, math.tau)
        self.init_beta = _clamp_float(_finite_float(self.init_beta, "init_beta"), 0.0, math.tau)

        max_entities_per_subqubo = max(1, math.ceil(self.num_processes / settings.public_max_bundles))
        min_qubit_max = max(self.num_cores, max_entities_per_subqubo * self.num_cores)
        requested_qubit_max = self.qubit_max if self.qubit_max is not None else self.num_cores * 4
        public_max_qubits = max(self.num_cores, settings.public_max_qubits)
        self.qubit_max = min(
            public_max_qubits,
            max(min_qubit_max, int(requested_qubit_max)),
        )
        if self.effective_num_bundles > settings.public_max_bundles:
            raise ValueError(
                "Configuration requires too many decomposition bundles for the "
                "public limits. Increase qubit_max or reduce num_processes."
            )

        self.io_alpha = _clamp_float(_finite_float(self.io_alpha, "io_alpha"), 0.0, 1.0)
        self.affinity_alpha = _clamp_float(_finite_float(self.affinity_alpha, "affinity_alpha"), 0.0, 1.0)
        self.homogeneity_threshold = _clamp_float(
            _finite_float(self.homogeneity_threshold, "homogeneity_threshold"), 0.0, 10.0
        )
        self.zscore_threshold = _clamp_float(_finite_float(self.zscore_threshold, "zscore_threshold"), 0.0, 10.0)

        self.min_rss = max(0.0, _finite_float(self.min_rss, "min_rss"))
        self.min_cpu = max(0.0, _finite_float(self.min_cpu, "min_cpu"))
        self.cpu_interval = _clamp_int(int(self.cpu_interval), 1, 60)
        self.num_samples = _clamp_int(int(self.num_samples), 1, 20)
        return self

    @property
    def effective_num_bundles(self) -> int:
        max_per_bundle = max(1, int(self.qubit_max or self.num_cores) // self.num_cores)
        return min(math.ceil(self.num_processes / max_per_bundle), self.num_processes)
