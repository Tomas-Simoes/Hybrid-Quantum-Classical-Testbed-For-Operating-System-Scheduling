from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import numpy as np

from decomposition.subqubo_heuristics import Heuristic

@dataclass 
class WorkloadEntity:
    entity_id: int 
    cpu_weight: float 
    rss_mb: float 
    label: str 
    def __str__(self) -> str:
        return f"[{self.entity_id}] {self.label[:15]:<15} | CPU: {self.cpu_weight:>6.2%} | RAM: {self.rss_mb:>8.1f}MB"
    __repr__ = __str__

@dataclass 
class Workload:
    entities: List[WorkloadEntity]
    num_cores: int 
    snapshot_id: str
    fixed_assignments: Dict[int, int] = field(default_factory=dict)
    fixed_loads: Dict[int, float] = field(default_factory=dict)

    @property
    def total_weight(self) -> float:
        return sum(e.cpu_weight for e in self.entities) + sum(self.fixed_loads.values())

    @property
    def fixed_load_per_core(self) -> np.ndarray:
        phi = np.zeros(self.num_cores)
        for entity_id, core in self.fixed_assignments.items():
            if 0 <= core < self.num_cores:
                phi[core] += self.fixed_loads.get(entity_id, 0.0)
        return phi
    
    @property
    def entity_map(self) -> Dict[int, WorkloadEntity]:
        return {e.entity_id: e for e in self.entities}

    def get_entity(self, entity_id: int) -> WorkloadEntity:
        return self.entity_map.get(entity_id)
    
    def __str__(self) -> str:
        return (f"Workload Snapshot: {self.snapshot_id}\n"
                f"Entities: {len(self.entities)} | Cores: {self.num_cores} | "
                f"Total CPU: {self.total_weight:.2%}")
    __repr__ = __str__


# ---------------------------------------------------------------------------
# Tracer output
# ---------------------------------------------------------------------------
@dataclass
class ProcessInfo:
    pid: int
    command: str
    current_core: int
    cpu_weight: float
    rss_mb: float
    priority: int
    io_wait_ratio: float | None 
    priority_class: str | None 

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "command": self.command,
            "cpu_weight": self.cpu_weight,
            "current_core": self.current_core,
            "rss_mb": self.rss_mb,
            "priority": self.priority,
            "io_wait_ratio": self.io_wait_ratio,
            "priority_class": self.priority_class,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProcessInfo:
        return cls(
            pid=d["pid"],
            command=d["command"],
            cpu_weight=d["cpu_weight"],
            current_core=d["current_core"],
            rss_mb=d["rss_mb"],
            priority=d["priority"],
            io_wait_ratio=d.get("io_wait_ratio"),
            priority_class=d.get("priority_class"),
        )
    
@dataclass
class SystemSnapshot:
    processes: List[ProcessInfo]
    num_cores: int 
    total_ram_mb: int | None 
    snapshot_id: str | None
    timestamp: float 

    def to_workload(self) -> Workload:
        return Workload(
            entities=[WorkloadEntity(
                entity_id=p.pid,
                cpu_weight=p.cpu_weight,
                rss_mb=p.rss_mb,
                label=f"pid_{p.pid}"
            ) for p in self.processes
            ],
            num_cores=self.num_cores,
            snapshot_id=self.snapshot_id
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "num_cores": self.num_cores,
            "total_ram_mb": self.total_ram_mb,
            "processes": [p.to_dict() for p in self.processes],
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SystemSnapshot:
        return cls(
            timestamp=d["timestamp"],
            num_cores=d["num_cores"],
            processes=[ProcessInfo.from_dict(p) for p in d["processes"]],
            total_ram_mb=d.get("total_ram_mb"),
            snapshot_id=d["snapshot_id"],
        )
    
@dataclass
class SnapshotObject:
    snapshot: SystemSnapshot | None
    min_rss: float 
    min_weight: float 
    cpu_interval: float 
    num_samples: int
    
# ---------------------------------------------------------------------------
#  Decomposition Engine output
# ---------------------------------------------------------------------------
@dataclass 
class FeatureMatrix:
    F_norm: np.ndarray # The z-score normalized values
    pids: list         # PID mapping for index reference
    F: np.ndarray      # The original (corrected) weights
    w_eff: np.ndarray
@dataclass 
class AffinityMatrix:
    A: np.ndarray

@dataclass
class Bundle:
    bundle_id: int
    member_pids: List[int]
    aggregate_cpu_weight: float
    aggregate_rss_mb: float
    representative_cmd: str 

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "member_pids": self.member_pids,
            "aggregate_cpu_weight": self.aggregate_cpu_weight,
            "aggregate_rss_mb": self.aggregate_rss_mb,
            "representative_cmd": self.representative_cmd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Bundle:
        return cls(
            bundle_id=d["bundle_id"],
            member_pids=d["member_pids"],
            aggregate_cpu_weight=d["aggregate_cpu_weight"],
            aggregate_rss_mb=d["aggregate_rss_mb"],
            representative_cmd=d.get("representative_cmd", "mixed"),
        )
    
    def __str__(self) -> str:
        return (f"Bundle_{self.bundle_id:<2} | "
                f"Members: {len(self.member_pids):>2} | "
                f"CPU: {self.aggregate_cpu_weight:>6.2%} | "
                f"MEM: {self.aggregate_rss_mb:>8.1f}MB | "
                f"CMD: {self.representative_cmd}")
    
@dataclass
class ClusteredSnapshot:
    bundles: List[Bundle]
    num_cores: int
    source_snapshot_id: str
    rt_procs: List[ProcessInfo] = field(default_factory=list)

    def to_workload(self) -> Workload:
        return Workload(
            entities=[
                WorkloadEntity(
                    entity_id=b.bundle_id,  # negative to distinguish from real PIDs
                    cpu_weight=b.aggregate_cpu_weight,
                    rss_mb=b.aggregate_rss_mb,
                    label=f"Bundle_{b.bundle_id}"
                )
                for b in self.bundles
            ],
            num_cores=self.num_cores,
            snapshot_id=self.source_snapshot_id,
            fixed_assignments={p.pid: p.current_core for p in self.rt_procs},
            fixed_loads={p.pid: p.cpu_weight for p in self.rt_procs},
        )
    def to_dict(self) -> dict:
        return {
            "bundles": [c.to_dict() for c in self.bundles],
            "num_cores": self.num_cores,
            "source_snapshot_id": self.source_snapshot_id,
            "rt_procs": [p.to_dict() for p in self.rt_procs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClusteredSnapshot:
        return cls(
            bundles=[Bundle.from_dict(c) for c in d["bundles"]],
            num_cores=d["num_cores"],
            source_snapshot_id=d["source_snapshot_id"],
            rt_procs=[ProcessInfo.from_dict(p) for p in d.get("rt_procs", [])],
        )
# ---------------------------------------------------------------------------
# Translator output
# ---------------------------------------------------------------------------
@dataclass
class QUBOInstance:
    Q: np.ndarray                              # shape (num_variables, num_variables)
    num_variables: int
    variable_map: Dict[int, Tuple[int, int]]   # var_index -> (entity_id, core_id)
    num_entities: int
    num_cores: int
    penalty_weight: float
    iteration_index: int
    source_snapshot_id: str

    def to_dict(self) -> dict:
        return {
            "Q": self.Q.tolist(),
            "num_variables": self.num_variables,
            # JSON requires string keys; tuple becomes a 2-element list
            "variable_map": {str(k): list(v) for k, v in self.variable_map.items()},
            "num_entities": self.num_entities,
            "num_cores": self.num_cores,
            "penalty_weight": self.penalty_weight,
            "iteration_index": self.iteration_index,
            "source_snapshot_id": self.source_snapshot_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QUBOInstance:
        return cls(
            Q=np.array(d["Q"]),
            num_variables=d["num_variables"],
            variable_map={int(k): tuple(v) for k, v in d["variable_map"].items()},
            num_entities=d["num_entities"],
            num_cores=d["num_cores"],
            penalty_weight=d["penalty_weight"],
            iteration_index=d["iteration_index"],
            source_snapshot_id=d["source_snapshot_id"],
        )
# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------
@dataclass
class SolverResult:
    bitstring: np.ndarray                      # 1-D binary array, length num_variables
    decoded_assignments: Dict[int, int]        # entity_id -> core_id
    energy: float
    is_feasible: bool
    solver_backend: str                        # "brute_force" | "simulated_annealing" | "qaoa" | …
    solve_time_ms: float
    solver_params: Dict = field(default_factory=dict)
    probs: np.ndarray = None
    convergence_curve: list = None
    
    def to_dict(self) -> dict:
        return {
            "bitstring": self.bitstring.tolist(),
            "decoded_assignments": {str(k): v for k, v in self.decoded_assignments.items()},
            "energy": self.energy,
            "is_feasible": self.is_feasible,
            "solver_backend": self.solver_backend,
            "solve_time_ms": self.solve_time_ms,
            "solver_params": self.solver_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SolverResult:
        return cls(
            bitstring=np.array(d["bitstring"]),
            decoded_assignments={int(k): v for k, v in d["decoded_assignments"].items()},
            energy=d["energy"],
            is_feasible=d["is_feasible"],
            solver_backend=d["solver_backend"],
            solve_time_ms=d["solve_time_ms"],
            solver_params=d.get("solver_params", {}),
        )
@dataclass
class PipelineResult:
    iterations: List[SolverResult]
    final_assignments: Dict[int, int]          # pid -> core_id
    total_solve_time_ms: float
    num_iterations: int
    source_snapshot_id: str

    def to_dict(self) -> dict:
        return {
            "iterations": [r.to_dict() for r in self.iterations],
            "final_assignments": {str(k): v for k, v in self.final_assignments.items()},
            "total_solve_time_ms": self.total_solve_time_ms,
            "num_iterations": self.num_iterations,
            "source_snapshot_id": self.source_snapshot_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PipelineResult:
        return cls(
            iterations=[SolverResult.from_dict(r) for r in d["iterations"]],
            final_assignments={int(k): v for k, v in d["final_assignments"].items()},
            total_solve_time_ms=d["total_solve_time_ms"],
            num_iterations=d["num_iterations"],
            source_snapshot_id=d["source_snapshot_id"],
        )
@dataclass
class SchedulingOutput:
    result: SolverResult
    validation: dict
    used_snapshot: SystemSnapshot
    alpha: float
    qubo_instance: QUBOInstance
    qaoa_cfg: QAOAConfig
    qubo_cfg: QUBOConfig

    def __iter__(self):
        return iter((
            self.result, 
            self.validation, 
            self.used_snapshot,
            self.alpha,
            self.qubo_instance, 
            self.qaoa_cfg, 
            self.qubo_cfg
        ))
    


@dataclass
class IterativeSchedulingOutput:
    # Core results
    final_assignments: Dict[int, int]       # entity_id -> core
    solver_results: List[SolverResult]      # one per sub-QUBO
    phi_history: List[np.ndarray]           # phi vector after each sub-QUBO

    # Context
    used_workload: Workload
    qubo_instance: QUBOInstance             # global Q used for final validation
    qaoa_cfg: QAOAConfig
    qubo_cfg: QUBOConfig
    global_result: SolverResult | None = None
    validation: dict | None = None
    alpha: float | None = None

    # Derived metrics — computed in __post_init__
    final_phi: np.ndarray = field(init=False)
    L_avg: float = field(init=False)
    load_imbalance: float = field(init=False)
    total_solve_time_ms: float = field(init=False)
    num_sub_qubos: int = field(init=False)
    num_feasible: int = field(init=False)
    all_feasible: bool = field(init=False)

    def __post_init__(self):
        self.final_phi          = self.phi_history[-1]
        self.L_avg              = self.used_workload.total_weight / self.used_workload.num_cores
        self.load_imbalance     = float(self.final_phi.max() - self.final_phi.min())
        self.total_solve_time_ms = sum(r.solve_time_ms for r in self.solver_results)
        self.num_sub_qubos      = len(self.solver_results)
        self.num_feasible       = sum(1 for r in self.solver_results if r.is_feasible)
        self.all_feasible       = self.num_feasible == self.num_sub_qubos
# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
@dataclass
class QUBOConfig:
    penalty: float
    num_cores: int
    snapshot: SystemSnapshot | None
    target_load: float | None

@dataclass
class QAOAConfig:
    layers: int
    steps: int
    learning_rate: float
    top_k: int
    mixer_type: Literal["xy", "x"] = "xy"
    init_gamma: float = 0.5
    init_beta: float = 0.5

    def __post_init__(self):
        if self.mixer_type not in ("xy", "x"):
            raise ValueError(f"Unsupported mixer_type '{self.mixer_type}'. Expected 'xy' or 'x'.")

@dataclass 
class TracerConfig:
    min_rss: float
    min_cpu: float
    cpu_interval: int 
    num_samples: int
    live_mode: bool

@dataclass
class DecompositorConfig:
    qubit_max: int
    num_cores: int         
    io_alpha: float
    affinity_alpha: float
    homogeneity_threshold: float
    zscore_threshold: float
    sorting_strategy: Heuristic = Heuristic.WEIGHT_DESCENDING

    def __post_init__(self):
        if self.qubit_max < self.num_cores:
            raise ValueError("qubit_max must be greater than or equal to num_cores.")

    def num_bundles(self, n_processes: int) -> int:
        max_per_bundle = self.qubit_max // self.num_cores   
        return min(math.ceil(n_processes / max_per_bundle), n_processes)
# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------
def round_trip_test() -> None:
    snapshot_id = "test-snapshot-uuid-1234"

    # ProcessInfo
    proc = ProcessInfo(
        pid=42,
        command="python3",
        cpu_weight=0.8,
        current_core=1,
        rss_mb=128.5,
        priority=20,
        io_wait_ratio=0.1,
        priority_class="BE",
    )
    assert ProcessInfo.from_dict(proc.to_dict()) == proc

    # SystemSnapshot (auto-generated id overridden for determinism)
    snap = SystemSnapshot(
        timestamp=1_700_000_000.0,
        num_cores=4,
        processes=[proc],
        total_ram_mb=16384.0,
        snapshot_id=snapshot_id,
    )
    snap_rt = SystemSnapshot.from_dict(snap.to_dict())
    assert snap_rt == snap

    # Bundle / ClusteredSnapshot
    bundle = Bundle(
        bundle_id=0,
        member_pids=[42, 99],
        aggregate_cpu_weight=1.3,
        aggregate_rss_mb=256.0,
        representative_cmd="python",
    )
    assert Bundle.from_dict(bundle.to_dict()) == bundle

    clustered = ClusteredSnapshot(
        bundles=[bundle],
        num_cores=4,
        source_snapshot_id=snapshot_id,
        rt_procs=[proc],
    )
    assert ClusteredSnapshot.from_dict(clustered.to_dict()) == clustered

    # QUBOInstance
    Q = np.array([[1.0, -0.5], [-0.5, 1.0]])
    qubo = QUBOInstance(
        Q=Q,
        num_variables=2,
        variable_map={0: (42, 0), 1: (42, 1)},
        num_entities=1,
        num_cores=2,
        penalty_weight=9.0,
        iteration_index=0,
        source_snapshot_id=snapshot_id,
    )
    qubo_rt = QUBOInstance.from_dict(qubo.to_dict())
    assert np.array_equal(qubo_rt.Q, qubo.Q)
    assert qubo_rt.num_variables == qubo.num_variables
    assert qubo_rt.variable_map == qubo.variable_map
    assert qubo_rt.num_entities == qubo.num_entities
    assert qubo_rt.num_cores == qubo.num_cores
    assert qubo_rt.penalty_weight == qubo.penalty_weight
    assert qubo_rt.iteration_index == qubo.iteration_index
    assert qubo_rt.source_snapshot_id == qubo.source_snapshot_id

    # SolverResult
    result = SolverResult(
        bitstring=np.array([1, 0]),
        decoded_assignments={42: 0},
        energy=-1.25,
        is_feasible=True,
        solver_backend="brute_force",
        solve_time_ms=3.7,
        solver_params={"penalty": 9.0},
    )
    result_rt = SolverResult.from_dict(result.to_dict())
    assert np.array_equal(result_rt.bitstring, result.bitstring)
    assert result_rt.decoded_assignments == result.decoded_assignments
    assert result_rt.energy == result.energy
    assert result_rt.is_feasible == result.is_feasible
    assert result_rt.solver_backend == result.solver_backend
    assert result_rt.solve_time_ms == result.solve_time_ms
    assert result_rt.solver_params == result.solver_params

    # PipelineResult
    pipeline = PipelineResult(
        iterations=[result],
        final_assignments={42: 0, 99: 1},
        total_solve_time_ms=3.7,
        num_iterations=1,
        source_snapshot_id=snapshot_id,
    )
    pipeline_rt = PipelineResult.from_dict(pipeline.to_dict())
    assert pipeline_rt.final_assignments == pipeline.final_assignments
    assert pipeline_rt.total_solve_time_ms == pipeline.total_solve_time_ms
    assert pipeline_rt.num_iterations == pipeline.num_iterations
    assert pipeline_rt.source_snapshot_id == pipeline.source_snapshot_id
    assert len(pipeline_rt.iterations) == 1

    # Verify full JSON round-trip goes through json.dumps/loads without error
    for obj in [proc, snap, bundle, clustered, qubo, result, pipeline]:
        json.dumps(obj.to_dict())

    print("All round-trip assertions passed.")


if __name__ == "__main__":
    round_trip_test()
