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
from data_contracts import QUBOConfig, Workload, WorkloadEntity
from solver.brute_force_solver import BruteForceSolver


def tiny_workload() -> Workload:
    return Workload(
        entities=[
            WorkloadEntity(entity_id=101, cpu_weight=0.25, rss_mb=32.0, label="proc-a"),
            WorkloadEntity(entity_id=202, cpu_weight=0.50, rss_mb=64.0, label="proc-b"),
            WorkloadEntity(entity_id=303, cpu_weight=0.75, rss_mb=96.0, label="proc-c"),
        ],
        num_cores=2,
        snapshot_id="tiny-static-snapshot",
    )


def two_process_workload() -> Workload:
    return Workload(
        entities=[
            WorkloadEntity(entity_id=101, cpu_weight=0.40, rss_mb=32.0, label="proc-a"),
            WorkloadEntity(entity_id=202, cpu_weight=0.60, rss_mb=64.0, label="proc-b"),
        ],
        num_cores=2,
        snapshot_id="two-process-snapshot",
    )


def qubo_config(penalty: float = 5.0) -> QUBOConfig:
    return QUBOConfig(
        penalty=penalty,
        num_cores=2,
        snapshot=None,
        target_load=None,
    )


class CoreAssignmentBuilderTests(unittest.TestCase):
    def test_build_populates_qubo_instance_metadata(self) -> None:
        workload = tiny_workload()
        qubo = CoreAssignmentBuilder(qubo_config()).build(workload)

        expected_num_variables = len(workload.entities) * workload.num_cores

        self.assertEqual(qubo.num_variables, expected_num_variables)
        self.assertEqual(qubo.Q.shape, (expected_num_variables, expected_num_variables))
        self.assertEqual(qubo.num_entities, len(workload.entities))
        self.assertEqual(qubo.num_cores, workload.num_cores)
        self.assertEqual(qubo.penalty_weight, 5.0)
        self.assertEqual(qubo.iteration_index, 0)
        self.assertEqual(qubo.source_snapshot_id, workload.snapshot_id)

    def test_variable_map_contains_each_process_core_pair_once(self) -> None:
        workload = tiny_workload()
        qubo = CoreAssignmentBuilder(qubo_config()).build(workload)

        expected_variable_map = {
            0: (101, 0),
            1: (101, 1),
            2: (202, 0),
            3: (202, 1),
            4: (303, 0),
            5: (303, 1),
        }

        self.assertEqual(qubo.variable_map, expected_variable_map)
        self.assertEqual(set(qubo.variable_map), set(range(qubo.num_variables)))
        self.assertEqual(len(set(qubo.variable_map.values())), qubo.num_variables)

    def test_q_matrix_matches_expected_terms_for_tiny_workload(self) -> None:
        qubo = CoreAssignmentBuilder(qubo_config(penalty=3.0)).build(two_process_workload())

        expected_q = np.array(
            [
                [-3.24, 3.00, 0.24, 0.00],
                [3.00, -3.24, 0.00, 0.24],
                [0.24, 0.00, -3.24, 3.00],
                [0.00, 0.24, 3.00, -3.24],
            ]
        )

        np.testing.assert_allclose(qubo.Q, expected_q)

    def test_one_hot_bitstring_decodes_to_feasible_assignments(self) -> None:
        workload = tiny_workload()
        qubo = CoreAssignmentBuilder(qubo_config()).build(workload)
        expected_assignments = {101: 1, 202: 0, 303: 1}
        bitstring = np.zeros(qubo.num_variables, dtype=int)

        for idx, (entity_id, core_id) in qubo.variable_map.items():
            if expected_assignments[entity_id] == core_id:
                bitstring[idx] = 1

        decoded_assignments, is_feasible = BruteForceSolver._decode_assignments(
            bitstring,
            qubo,
        )

        self.assertTrue(is_feasible)
        self.assertEqual(decoded_assignments, expected_assignments)
        self.assertEqual(
            set(decoded_assignments),
            {entity.entity_id for entity in workload.entities},
        )

    def test_non_one_hot_bitstring_decodes_as_infeasible(self) -> None:
        qubo = CoreAssignmentBuilder(qubo_config()).build(tiny_workload())
        bitstring = np.array([1, 1, 0, 1, 1, 0])

        decoded_assignments, is_feasible = BruteForceSolver._decode_assignments(
            bitstring,
            qubo,
        )

        self.assertFalse(is_feasible)
        self.assertNotIn(101, decoded_assignments)


if __name__ == "__main__":
    unittest.main()
