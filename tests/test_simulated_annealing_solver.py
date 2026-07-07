from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from builder.builder_core import CoreAssignmentBuilder
from data_contracts import QUBOConfig, SolverResult, Workload, WorkloadEntity
from experiments.investigative_runtime import AnnealingSolver, InvestigativeValidator


def workload(num_entities: int = 4, num_cores: int = 2) -> Workload:
    return Workload(
        entities=[
            WorkloadEntity(
                entity_id=100 + i,
                cpu_weight=0.1 + 0.03 * i,
                rss_mb=32.0 + i,
                label=f"proc-{i}",
            )
            for i in range(num_entities)
        ],
        num_cores=num_cores,
        snapshot_id="annealing-test-snapshot",
    )


def qubo_config(num_cores: int = 2) -> QUBOConfig:
    return QUBOConfig(
        penalty=5.0,
        num_cores=num_cores,
        snapshot=None,
        target_load=None,
    )


class AnnealingSolverTests(unittest.TestCase):
    def test_solve_returns_feasible_one_hot_assignment(self) -> None:
        qubo = CoreAssignmentBuilder(qubo_config()).build(workload())
        result = AnnealingSolver(sweeps=100, restarts=4, seed=7).solve(qubo)

        self.assertTrue(result.is_feasible)
        self.assertEqual(result.solver_backend, "simulated_annealing_one_hot")
        self.assertEqual(len(result.decoded_assignments), qubo.num_entities)
        self.assertEqual(len(result.bitstring), qubo.num_variables)
        self.assertAlmostEqual(result.energy, float(result.bitstring.T @ qubo.Q @ result.bitstring))

        for entity_index in range(qubo.num_entities):
            start = entity_index * qubo.num_cores
            stop = start + qubo.num_cores
            self.assertEqual(result.bitstring[start:stop].sum(), 1)

    def test_validator_uses_annealing_when_brute_force_is_too_large(self) -> None:
        qubo = CoreAssignmentBuilder(qubo_config()).build(workload(num_entities=12))
        bitstring = np.zeros(qubo.num_variables, dtype=int)
        for entity_index in range(qubo.num_entities):
            bitstring[entity_index * qubo.num_cores] = 1

        candidate = SolverResult(
            bitstring=bitstring,
            decoded_assignments={
                entity_id: core
                for _var_idx, (entity_id, core) in qubo.variable_map.items()
                if core == 0
            },
            energy=float(bitstring.T @ qubo.Q @ bitstring),
            is_feasible=True,
            solver_backend="test_candidate",
            solve_time_ms=0.0,
        )

        validation = InvestigativeValidator().validate(qubo, candidate)

        self.assertIsNotNone(validation["brute_force_error"])
        self.assertIsNone(validation["global_energy"])
        self.assertIsNotNone(validation["annealing_energy"])
        self.assertTrue(validation["annealing_is_feasible"])
        self.assertEqual(validation["annealing_backend"], "simulated_annealing_one_hot")
        self.assertIsNotNone(validation["annealing_gap"])


if __name__ == "__main__":
    unittest.main()
