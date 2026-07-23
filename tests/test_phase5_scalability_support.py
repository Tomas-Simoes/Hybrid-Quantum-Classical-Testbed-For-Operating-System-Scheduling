from __future__ import annotations

import csv
import sys
import tempfile
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
from data_contracts import QAOAConfig, QUBOConfig, SolverResult, Workload, WorkloadEntity
from investigative_runtime import (
    FeasibleBruteForceSolver,
    InvestigativeValidationConfig,
    InvestigativeValidator,
    _offset_free_quality_metrics,
    assignment_balance_metrics,
)
from scenario_runner import build_generated_weights, load_toml
from solver.brute_force_solver import BruteForceSolver
from solver.pennylane_solver import PennylaneSolver
from sweep_runner import (
    load_reused_scalability_records,
    prepare_repeat_variant,
    repeat_count_for_variant,
    run_adaptive_sweep_scenario,
)


def build_qubo(weights: list[float]):
    workload = Workload(
        entities=[
            WorkloadEntity(100 + i, weight, 32.0, f"p{i}")
            for i, weight in enumerate(weights)
        ],
        num_cores=2,
        snapshot_id="phase5-test",
    )
    return CoreAssignmentBuilder(QUBOConfig(5.0, 2, None, None)).build(workload)


class Phase5ScalabilitySupportTests(unittest.TestCase):
    def test_production_solver_limits_candidates_to_configured_top_k(self) -> None:
        solver = PennylaneSolver(QAOAConfig(2, 100, 0.1, top_k=3))
        selected = solver._candidate_indices([0.1, 0.4, 0.2, 0.3, 0.0])

        self.assertEqual(selected.tolist(), [1, 3, 2])

    def test_render_runtime_skips_gpu_probe(self) -> None:
        solver = PennylaneSolver(QAOAConfig(1, 1, 0.1, top_k=1))

        with patch.dict("os.environ", {"RENDER": "true"}, clear=True):
            with patch.object(solver, "_solve_on_device", return_value="cpu") as solve:
                result = solver.solve("qubo")

        self.assertEqual(result, "cpu")
        solve.assert_called_once_with("qubo", "lightning.qubit")

    def test_force_cpu_env_skips_gpu_probe(self) -> None:
        solver = PennylaneSolver(QAOAConfig(1, 1, 0.1, top_k=1))

        with patch.dict("os.environ", {"QAOA_FORCE_CPU": "true"}, clear=True):
            with patch.object(solver, "_solve_on_device", return_value="cpu") as solve:
                result = solver.solve("qubo")

        self.assertEqual(result, "cpu")
        solve.assert_called_once_with("qubo", "lightning.qubit")

    def test_explicit_force_cpu_false_allows_gpu_first_on_render(self) -> None:
        solver = PennylaneSolver(QAOAConfig(1, 1, 0.1, top_k=1))

        with patch.dict(
            "os.environ",
            {"RENDER": "true", "QAOA_FORCE_CPU": "false"},
            clear=True,
        ):
            with patch.object(solver, "_solve_on_device", return_value="gpu") as solve:
                result = solver.solve("qubo")

        self.assertEqual(result, "gpu")
        solve.assert_called_once_with("qubo", "lightning.gpu")

    def test_uniform_random_instances_are_seeded_and_strictly_positive(self) -> None:
        config = {
            "num_processes": 8,
            "weight_strategy": "uniform_random",
            "instance_seed": 17,
        }
        first = build_generated_weights(config, {}, [])
        second = build_generated_weights(config, {}, [])
        different = build_generated_weights({**config, "instance_seed": 18}, {}, [])

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertTrue(all(0.0 < weight <= 1.0 for weight in first))

    def test_feasible_bruteforce_matches_legacy_certified_energy(self) -> None:
        qubo = build_qubo([0.11, 0.19, 0.27, 0.43])

        legacy = BruteForceSolver().solve(qubo)
        feasible = FeasibleBruteForceSolver(max_entities=20).solve(qubo)

        self.assertAlmostEqual(feasible.energy, legacy.energy)
        self.assertTrue(feasible.is_feasible)
        self.assertEqual(feasible.solver_params["assignments_evaluated"], 16)

    def test_auto_validator_uses_feasible_exact_baseline(self) -> None:
        qubo = build_qubo([0.1, 0.2, 0.3, 0.4])
        optimum = FeasibleBruteForceSolver().solve(qubo)
        candidate = SolverResult(
            bitstring=np.copy(optimum.bitstring),
            decoded_assignments=dict(optimum.decoded_assignments),
            energy=optimum.energy,
            is_feasible=True,
            solver_backend="test",
            solve_time_ms=0.0,
        )

        validation = InvestigativeValidator(
            InvestigativeValidationConfig(
                baseline_method="auto",
                brute_force_max_n=20,
            )
        ).validate(qubo, candidate)

        self.assertEqual(validation["baseline_method"], "bruteforce")
        self.assertTrue(validation["baseline_certified"])
        self.assertTrue(validation["is_optimal"])
        self.assertEqual(validation["relative_gap"], 0.0)

    def test_offset_free_metrics_use_loads_instead_of_shifted_energy(self) -> None:
        workload = Workload(
            entities=[
                WorkloadEntity(100 + index, weight, 32.0, f"p{index}")
                for index, weight in enumerate([0.4, 0.3, 0.2, 0.1])
            ],
            num_cores=2,
            snapshot_id="offset-free-test",
        )
        candidate = assignment_balance_metrics(
            workload,
            {100: 0, 101: 1, 102: 0, 103: 1},
        )

        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(candidate["load_imbalance"], 0.2)
        self.assertAlmostEqual(candidate["normalized_load_imbalance"], 0.4)
        self.assertAlmostEqual(candidate["load_balance_objective"], 0.02)

        comparison = _offset_free_quality_metrics(
            workload,
            {100: 0, 101: 1, 102: 0, 103: 1},
            {
                "baseline_method": "simulated_annealing",
                "annealing_assignments": {100: 0, 101: 1, 102: 1, 103: 0},
                "baseline_certified": False,
                "optimality_atol": 1e-9,
                "optimality_rtol": 1e-9,
            },
        )

        self.assertAlmostEqual(comparison["baseline_load_balance_objective"], 0.0)
        self.assertAlmostEqual(comparison["objective_regret"], 0.02)
        self.assertAlmostEqual(comparison["delta_imbalance"], 0.2)
        self.assertAlmostEqual(comparison["gap_estavel"], 0.04)
        self.assertFalse(comparison["baseline_match_offset_free"])
        self.assertFalse(comparison["is_optimal"])
        self.assertFalse(comparison["gap_relativo"] == 0.0)
        self.assertIsNone(comparison["certified_optimal_offset_free"])

    def test_stable_gap_uses_total_weight_not_near_zero_reference(self) -> None:
        pipeline = 0.002442
        reference = 0.000002825
        total_weight = 5.0
        stable_gap = (pipeline**2 - reference**2) / total_weight**2

        self.assertAlmostEqual(stable_gap, 2.38534240775e-7)
        self.assertLess(stable_gap, 1e-6)

    def test_problematic_imbalance_pair_is_not_an_optimal_match(self) -> None:
        pipeline_imbalance = 0.002442
        reference_imbalance = 0.000002825
        pipeline_objective = pipeline_imbalance**2 / 2.0
        reference_objective = reference_imbalance**2 / 2.0
        relative_gap = (pipeline_objective - reference_objective) / max(
            reference_objective, 1e-9
        )

        self.assertAlmostEqual(relative_gap, 2981.6780096875004, places=9)
        self.assertGreater(
            pipeline_objective - reference_objective,
            1e-9 + 1e-9 * reference_objective,
        )

    def test_repeat_tiers_and_independent_seed_streams(self) -> None:
        scenario = {
            "workload": {"num_processes": 65},
            "qaoa": {},
            "validation": {},
            "execution": {
                "repeat_tiers": [
                    {"max_n": 50, "repeats": 20},
                    {"max_n": 100, "repeats": 12},
                ]
            },
            "seeds": {"instance_seed_base": 0, "qaoa_seed_base": 10000},
        }

        self.assertEqual(repeat_count_for_variant(scenario), 12)
        seeded = prepare_repeat_variant(scenario, 3)
        self.assertEqual(seeded["workload"]["instance_seed"], 3)
        self.assertEqual(seeded["qaoa"]["random_seed"], 10003)
        self.assertEqual(seeded["validation"]["annealing_seed"], 3)

    def test_phase5_scenario_contains_requested_grid(self) -> None:
        scenario = load_toml(
            EXPERIMENTS_ROOT / "scenarios" / "phase5_e5_scalability_n.toml"
        )
        self.assertEqual(scenario["decomposition"]["qubit_max"], 8)
        self.assertEqual(scenario["qaoa"]["top_k"], 10)
        self.assertEqual(
            scenario["sweep"]["axes"][0]["values"],
            [20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200],
        )
        self.assertFalse(scenario["sweep"]["adaptive"]["enabled"])
        self.assertNotIn("resume", scenario["sweep"])
        self.assertEqual(
            scenario["sweep"]["adaptive"]["quality_metric"],
            "delta_imbalance",
        )
        self.assertNotIn("quality_ceiling_mean", scenario["sweep"]["adaptive"])
        self.assertNotIn("gap_ceiling_mean", scenario["sweep"]["adaptive"])
        self.assertNotIn("quality_threshold", scenario["sweep"]["adaptive"])

    def test_deleted_legacy_records_are_not_reused(self) -> None:
        scenario_path = (
            EXPERIMENTS_ROOT / "scenarios" / "phase5_e5_scalability_n.toml"
        )
        records = load_reused_scalability_records(
            scenario_path,
            load_toml(scenario_path),
            EXPERIMENTS_ROOT / "results",
        )

        self.assertEqual(records, [])

    def test_continuous_gap_signal_refines_and_runs_post_transition_points(self) -> None:
        scenario = {
            "id": "adaptive-test",
            "name": "Adaptive test",
            "workload": {"num_processes": 10},
            "execution": {"repeats": 1},
            "sweep": {
                "axes": [
                    {
                        "name": "num_processes",
                        "path": "workload.num_processes",
                        "values": [20, 30, 40, 50, 60, 70],
                    }
                ],
                "adaptive": {
                    "enabled": True,
                    "axis": "num_processes",
                    "quality_metric": "normalized_load_imbalance",
                    "quality_ceiling_mean": 0.001,
                    "monotonic_window": 3,
                    "monotonic_min_delta": 1e-6,
                    "refinement_points": 2,
                    "extra_points_after_transition": 3,
                    "max_mean_run_seconds": 180.0,
                    "tolerance_tiers": [1e-3, 1e-4, 1e-5],
                },
            },
        }
        gap_by_n = {
            20: 1e-5,
            30: 2e-5,
            33: 2.3e-5,
            37: 3.5e-5,
            40: 4e-5,
            50: 5e-5,
            60: 6e-5,
            70: 7e-5,
        }
        executed = []

        def fake_run(_path, variant, *_args):
            n = variant["workload"]["num_processes"]
            executed.append(n)
            return [
                {
                    "status": "success",
                    "duration_s": 1.0,
                    "config": variant,
                    "metrics": {
                        "is_optimal": False,
                        "gap_relativo": gap_by_n[n],
                        "load_imbalance": gap_by_n[n] * 2,
                        "normalized_load_imbalance": gap_by_n[n],
                        "load_balance_objective": gap_by_n[n] ** 2,
                        "baseline_load_balance_objective": 0.0,
                        "objective_regret": gap_by_n[n] ** 2,
                        "baseline_match_offset_free": False,
                        "certified_optimal_offset_free": False,
                        "tempo_total_ms": 1000.0,
                        "tempo_qaoa_ms": 990.0,
                        "tempo_overhead_ms": 10.0,
                        "num_sub_qubos": n / 4,
                        "baseline_method": "bruteforce",
                        "baseline_certified": True,
                    },
                }
            ]

        with tempfile.TemporaryDirectory() as tmp, patch(
            "sweep_runner.run_variant_repeats", side_effect=fake_run
        ):
            root = Path(tmp)
            records = run_adaptive_sweep_scenario(
                scenario_path=root / "adaptive.toml",
                scenario=scenario,
                results_dir=root / "results",
                aggregate_path=root / "aggregate.jsonl",
                cumulative_path=root / "all.jsonl",
                run_id="test-run",
                max_variants=None,
                dry_run=False,
            )
            self.assertTrue(
                (root / "results/adaptive_result/test-run_raw.csv").exists()
            )
            with (
                root / "results/adaptive_result/test-run_summary.csv"
            ).open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))

        self.assertEqual(executed, [20, 30, 40, 33, 37, 50, 60, 70])
        self.assertEqual(len(records), len(executed))
        self.assertIn("gap_relativo_median", summary_rows[0])
        self.assertIn("load_imbalance_mean", summary_rows[0])
        self.assertIn("normalized_load_imbalance_mean", summary_rows[0])
        self.assertIn("objective_regret_mean", summary_rows[0])
        self.assertIn("within_gap_1e-03_percent", summary_rows[0])


if __name__ == "__main__":
    unittest.main()
