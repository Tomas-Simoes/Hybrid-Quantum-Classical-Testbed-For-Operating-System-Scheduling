from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "core"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from builder.builder_core import CoreAssignmentBuilder
from data_contracts import QUBOConfig, QUBOInstance, Workload, WorkloadEntity
from solver.brute_force_solver import BRUTE_FORCE_VAR_LIMIT, BruteForceSolver


def tiny_qubo() -> QUBOInstance:
    workload = Workload(
        entities=[
            WorkloadEntity(entity_id=101, cpu_weight=0.40, rss_mb=32.0, label="proc-a"),
            WorkloadEntity(entity_id=202, cpu_weight=0.60, rss_mb=64.0, label="proc-b"),
            WorkloadEntity(entity_id=303, cpu_weight=0.20, rss_mb=16.0, label="proc-c"),
        ],
        num_cores=2,
        snapshot_id="brute-force-tiny-snapshot",
    )
    qubo_cfg = QUBOConfig(
        penalty=5.0,
        num_cores=workload.num_cores,
        snapshot=None,
        target_load=None,
    )
    return CoreAssignmentBuilder(qubo_cfg).build(workload)


def bitstring_for_core_choices(qubo: QUBOInstance, core_choices: tuple[int, ...]) -> np.ndarray:
    bitstring = np.zeros(qubo.num_variables, dtype=int)
    for entity_index, core_id in enumerate(core_choices):
        bitstring[entity_index * qubo.num_cores + core_id] = 1
    return bitstring


def feasible_energies(qubo: QUBOInstance) -> list[float]:
    energies = []
    for core_choices in product(range(qubo.num_cores), repeat=qubo.num_entities):
        bitstring = bitstring_for_core_choices(qubo, core_choices)
        energies.append(float(bitstring.T @ qubo.Q @ bitstring))
    return energies


class BruteForceSolverTests(unittest.TestCase):
    def test_solve_returns_feasible_assignment_for_tiny_qubo(self) -> None:
        qubo = tiny_qubo()
        result = BruteForceSolver().solve(qubo)

        self.assertTrue(result.is_feasible)
        self.assertEqual(result.solver_backend, "brute_force")
        self.assertEqual(len(result.bitstring), qubo.num_variables)
        self.assertEqual(len(result.decoded_assignments), qubo.num_entities)
        self.assertEqual(
            set(result.decoded_assignments),
            {entity_id for entity_id, _core_id in qubo.variable_map.values()},
        )
        self.assertTrue(
            all(0 <= core_id < qubo.num_cores for core_id in result.decoded_assignments.values())
        )

    def test_best_feasible_solution_satisfies_one_hot_constraints(self) -> None:
        qubo = tiny_qubo()
        result = BruteForceSolver().solve(qubo)

        for entity_index in range(qubo.num_entities):
            start = entity_index * qubo.num_cores
            stop = start + qubo.num_cores
            group = result.bitstring[start:stop]

            self.assertEqual(group.sum(), 1)

            chosen_core_offset = int(np.argmax(group))
            entity_id, core_id = qubo.variable_map[start + chosen_core_offset]
            self.assertEqual(result.decoded_assignments[entity_id], core_id)

    def test_energy_matches_q_matrix_evaluation(self) -> None:
        qubo = tiny_qubo()
        result = BruteForceSolver().solve(qubo)

        expected_energy = float(result.bitstring.T @ qubo.Q @ result.bitstring)

        self.assertAlmostEqual(result.energy, expected_energy)

    def test_solve_returns_lowest_feasible_energy(self) -> None:
        qubo = tiny_qubo()
        result = BruteForceSolver().solve(qubo)

        self.assertAlmostEqual(result.energy, min(feasible_energies(qubo)))

    def test_solver_refuses_qubos_above_variable_limit(self) -> None:
        num_variables = BRUTE_FORCE_VAR_LIMIT + 1
        qubo = QUBOInstance(
            Q=np.zeros((num_variables, num_variables)),
            num_variables=num_variables,
            variable_map={idx: (1000 + idx, 0) for idx in range(num_variables)},
            num_entities=num_variables,
            num_cores=1,
            penalty_weight=1.0,
            iteration_index=0,
            source_snapshot_id="oversized-qubo",
        )

        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            BruteForceSolver().solve(qubo)


if __name__ == "__main__":
    unittest.main()
