from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

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
    io_wait_ratio: float
    priority: int

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "command": self.command,
            "cpu_weight": self.cpu_weight,
            "current_core": self.current_core,
            "rss_mb": self.rss_mb,
            "priority": self.priority,
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
        )
@dataclass
class SystemSnapshot:
    timestamp: float
    num_cores: int
    processes: List[ProcessInfo]
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "num_cores": self.num_cores,
            "processes": [p.to_dict() for p in self.processes],
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SystemSnapshot:
        return cls(
            timestamp=d["timestamp"],
            num_cores=d["num_cores"],
            processes=[ProcessInfo.from_dict(p) for p in d["processes"]],
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
    
@dataclass
class ClusteredSnapshot:
    bundles: List[Bundle]
    num_cores: int
    source_snapshot_id: str

    def to_dict(self) -> dict:
        return {
            "bundles": [c.to_dict() for c in self.bundles],
            "num_cores": self.num_cores,
            "source_snapshot_id": self.source_snapshot_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClusteredSnapshot:
        return cls(
            bundles=[Bundle.from_dict(c) for c in d["bundles"]],
            num_cores=d["num_cores"],
            source_snapshot_id=d["source_snapshot_id"],
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
    
# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
@dataclass
class QUBOConfig:
    penalty: float
    num_cores: int
    snapshot: SystemSnapshot | None

@dataclass
class QAOAConfig:
    layers: int
    steps: int
    learning_rate: float
    top_k: int

@dataclass 
class TracerConfig:
    min_rss: float
    min_cpu: float
    cpu_interval: int 
    num_samples: int
    live_mode: bool

@dataclass
class DecompositorConfig:
    num_bundles: int
    io_alpha: float
    affinity_alpha: float
    affinity_sigma: float
    homogeneity_threshold:  float

# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------
def round_trip_test() -> None:
    snapshot_id = "test-snapshot-uuid-1234"

    # ProcessInfo
    proc = ProcessInfo(pid=42, command="python3", cpu_weight=0.8, current_core=1, rss_mb=128.5, priority=20)
    assert ProcessInfo.from_dict(proc.to_dict()) == proc

    # SystemSnapshot (auto-generated id overridden for determinism)
    snap = SystemSnapshot(timestamp=1_700_000_000.0, num_cores=4, processes=[proc], snapshot_id=snapshot_id)
    snap_rt = SystemSnapshot.from_dict(snap.to_dict())
    assert snap_rt == snap

    # ClusterGroup
    cluster = ClusterGroup(
        cluster_id=0,
        member_pids=[42, 99],
        aggregate_cpu_weight=1.3,
        aggregate_rss_mb=256.0,
        contention_score=0.72,
    )
    assert ClusterGroup.from_dict(cluster.to_dict()) == cluster

    # DecomposedSubproblem
    subproblem = DecomposedSubproblem(
        clusters=[cluster],
        candidate_cores=[0, 1, 2],
        fixed_load_per_core={0: 0.4, 1: 0.6},
        iteration_index=0,
        source_snapshot_id=snapshot_id,
    )
    assert DecomposedSubproblem.from_dict(subproblem.to_dict()) == subproblem

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
    for obj in [proc, snap, cluster, subproblem, pipeline]:
        json.dumps(obj.to_dict())

    print("All round-trip assertions passed.")


if __name__ == "__main__":
    round_trip_test()
