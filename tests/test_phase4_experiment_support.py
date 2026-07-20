from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "core"
EXPERIMENTS_ROOT = SRC_ROOT / "experiments"
for path in (SRC_ROOT, EXPERIMENTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from builder.builder_core import CoreAssignmentBuilder
from data_contracts import QUBOConfig, SolverResult, Workload, WorkloadEntity
from investigative_runtime import (
    InvestigativeValidationConfig,
    InvestigativeValidator,
)
from scenario_runner import build_run_inputs
from sweep_runner import expand_sweep


class Phase4ExperimentSupportTests(unittest.TestCase):
    def test_sweep_cases_cross_product_with_axes(self) -> None:
        scenario = {
            "id": "cross",
            "name": "Cross product",
            "value": 0,
            "sweep": {
                "cases": [
                    {"name": "a", "values": {"value": 10}},
                    {"name": "b", "values": {"value": 20}},
                ],
                "axes": [
                    {"name": "seed", "path": "qaoa.random_seed", "values": [1, 2, 3]},
                ],
            },
        }

        variants = expand_sweep(scenario)

        self.assertEqual(len(variants), 6)
        self.assertEqual({variant["value"] for variant in variants}, {10, 20})
        self.assertEqual(
            {variant["qaoa"]["random_seed"] for variant in variants},
            {1, 2, 3},
        )
        self.assertTrue(all(variant["sweep_context"]["variant_count"] == 6 for variant in variants))

    def test_synthetic_cluster_mode_builds_deterministic_snapshot(self) -> None:
        scenario = {
            "workload": {
                "mode": "synthetic_cluster",
                "num_cores": 2,
                "processes": [
                    {
                        "pid": 10,
                        "command": "cpu",
                        "cpu_weight": 0.8,
                        "rss_mb": 64.0,
                        "io_wait_ratio": 0.0,
                    },
                    {
                        "pid": 20,
                        "command": "io",
                        "cpu_weight": 0.2,
                        "rss_mb": 256.0,
                        "io_wait_ratio": 0.9,
                    },
                ],
            },
            "qaoa": {"random_seed": 7},
            "decomposition": {"qubit_max": 2, "num_cores": 2},
            "validation": {"always_run_annealing": True},
            "execution": {"pipeline": "iterative", "visualization": False},
        }

        _, _, _, _, snapshot, _, run_cfg, validation_cfg = build_run_inputs(scenario)

        self.assertEqual([process.pid for process in snapshot.processes], [10, 20])
        self.assertTrue(run_cfg.cluster_preset_snapshot)
        self.assertEqual(run_cfg.pipeline_mode, "iterative")
        self.assertFalse(run_cfg.enable_visualization)
        self.assertTrue(validation_cfg.always_run_annealing)
        self.assertEqual(validation_cfg.annealing_seed, 7)

    def test_validator_can_run_annealing_alongside_brute_force(self) -> None:
        workload = Workload(
            entities=[
                WorkloadEntity(100 + index, weight, 32.0, f"p{index}")
                for index, weight in enumerate([0.2, 0.3, 0.5])
            ],
            num_cores=2,
            snapshot_id="phase4-validator",
        )
        qubo = CoreAssignmentBuilder(
            QUBOConfig(5.0, 2, None, None)
        ).build(workload)
        bitstring = np.array([1, 0, 0, 1, 1, 0])
        result = SolverResult(
            bitstring=bitstring,
            decoded_assignments={100: 0, 101: 1, 102: 0},
            energy=float(bitstring.T @ qubo.Q @ bitstring),
            is_feasible=True,
            solver_backend="test",
            solve_time_ms=0.0,
        )

        validation = InvestigativeValidator(
            InvestigativeValidationConfig(
                always_run_annealing=True,
                annealing_seed=9,
            )
        ).validate(qubo, result)

        self.assertIsNotNone(validation["global_energy"])
        self.assertIsNotNone(validation["brute_force_solve_time_ms"])
        self.assertIsNotNone(validation["annealing_energy"])
        self.assertEqual(validation["annealing_backend"], "simulated_annealing_one_hot")

    def test_validator_uses_strict_optimality_tolerance(self) -> None:
        workload = Workload(
            entities=[
                WorkloadEntity(100 + index, weight, 32.0, f"p{index}")
                for index, weight in enumerate([0.2, 0.3, 0.5])
            ],
            num_cores=2,
            snapshot_id="phase4-strict-optimality",
        )
        # A very large penalty reproduces the additive-offset problem: default
        # relative tolerances would incorrectly accept the distinct energy.
        qubo = CoreAssignmentBuilder(
            QUBOConfig(1_000_000.0, 2, None, None)
        ).build(workload)
        bitstring = np.array([1, 0, 0, 1, 1, 0])
        candidate_energy = float(bitstring.T @ qubo.Q @ bitstring)
        candidate = SolverResult(
            bitstring=bitstring,
            decoded_assignments={100: 0, 101: 1, 102: 0},
            energy=candidate_energy,
            is_feasible=True,
            solver_backend="test",
            solve_time_ms=0.0,
        )
        near_but_distinct_optimum = SolverResult(
            bitstring=bitstring,
            decoded_assignments=candidate.decoded_assignments,
            energy=candidate_energy - 5e-8,
            is_feasible=True,
            solver_backend="mock-brute-force",
            solve_time_ms=0.0,
        )

        self.assertTrue(np.isclose(candidate_energy, near_but_distinct_optimum.energy))
        with patch(
            "investigative_runtime.BruteForceSolver.solve",
            return_value=near_but_distinct_optimum,
        ):
            validation = InvestigativeValidator().validate(qubo, candidate)

        self.assertFalse(validation["is_optimal"])


if __name__ == "__main__":
    unittest.main()
